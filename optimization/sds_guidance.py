"""
SDS (Score Distillation Sampling) guidance from ModelScope text-to-video model.
Adapted from OmniPhysGS. Provides gradient signal from a pretrained video diffusion
model to optimize spring-mass physics parameters.
"""
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline, DDIMScheduler


class VideoSDSGuidance:
    """
    Video SDS guidance using ModelScope text-to-video (ali-vilab/text-to-video-ms-1.7b).
    """

    def __init__(self, pretrained_model='ali-vilab/text-to-video-ms-1.7b',
                 guidance_scale=100.0, min_step_percent=0.02, max_step_percent=0.98,
                 weighting='sds', device='cuda'):
        self.device = device
        self.guidance_scale = guidance_scale
        self.weighting = weighting

        print(f"[SDS] Loading {pretrained_model}...")
        pipe = DiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16)

        self.vae = pipe.vae.eval().to(device)
        self.unet = pipe.unet.eval().to(device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval().to(device)

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder='scheduler')
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.alphas = self.scheduler.alphas_cumprod.to(device)
        self.vae_scale = self.vae.config.scaling_factor

        print(f"[SDS] Ready. GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    @torch.no_grad()
    def encode_prompt(self, prompt, negative_prompt=''):
        """Encode text prompt to embeddings [2, 77, D]."""
        text_input = self.tokenizer(prompt, padding='max_length', max_length=77,
                                     truncation=True, return_tensors='pt')
        text_emb = self.text_encoder(text_input.input_ids.to(self.device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=77,
                                       truncation=True, return_tensors='pt')
        uncond_emb = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return torch.cat([uncond_emb, text_emb], dim=0)  # [2, 77, D]

    def encode_video(self, video_frames):
        """
        Encode video frames to latent space.
        Args:
            video_frames: [T, H, W, 3] in [0, 1] or [B, 3, T, H, W]
        Returns:
            latents: [1, 4, T, H/8, W/8]
        """
        if video_frames.dim() == 4 and video_frames.shape[-1] == 3:
            frames = video_frames.permute(0, 3, 1, 2)  # [T, 3, H, W]
        elif video_frames.dim() == 5:
            frames = video_frames.squeeze(0).permute(1, 0, 2, 3)  # [T, 3, H, W]
        else:
            frames = video_frames

        T = frames.shape[0]
        frames_norm = frames * 2.0 - 1.0

        latents_list = []
        with torch.no_grad():
            for i in range(T):
                lat = self.vae.encode(frames_norm[i:i+1].half()).latent_dist.sample()
                latents_list.append(lat)

        latents = torch.cat(latents_list, dim=0)  # [T, 4, H/8, W/8]
        latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 4, T, H/8, W/8]
        return latents * self.vae_scale

    def compute_sds_loss(self, video_frames, text_embeddings):
        """
        Compute SDS loss on rendered video frames.

        Args:
            video_frames: [T, H, W, 3] in [0, 1], rendered from simulation
            text_embeddings: [2, 77, D] from encode_prompt()

        Returns:
            loss_sds: scalar tensor with grad
        """
        # Resize to 256x256 for the model
        if video_frames.dim() == 4 and video_frames.shape[-1] == 3:
            frames = video_frames.permute(0, 3, 1, 2)  # [T, 3, H, W]
        else:
            frames = video_frames

        T = frames.shape[0]
        frames_resized = F.interpolate(frames, size=(256, 256), mode='bilinear', align_corners=False)

        # Encode to latents (with grad for backprop through rendering)
        frames_norm = frames_resized * 2.0 - 1.0
        latents_list = []
        for i in range(T):
            lat = self.vae.encode(frames_norm[i:i+1].half()).latent_dist.sample()
            latents_list.append(lat.float())
        latents = torch.cat(latents_list, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
        latents = latents * self.vae_scale

        # Random timestep
        t = torch.randint(self.min_step, self.max_step + 1, (1,), device=self.device, dtype=torch.long)

        # Add noise and predict
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_input = torch.cat([latents_noisy] * 2, dim=0).half()
            noise_pred = self.unet(
                latent_input, torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings.half(),
            ).sample.float()

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Weighting
        if self.weighting == 'sds':
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1, 1)
        elif self.weighting == 'fantasia3d':
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1, 1)
        else:
            w = 1.0

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction='sum')
        return loss_sds
