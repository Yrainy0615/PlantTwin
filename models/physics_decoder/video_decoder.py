"""
Feed-forward video physics decoder.
Adapted from ReconPhys (third_party/ReconPhys/internvit_predictor.py).
Predicts spring-mass physical parameters from input video.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != (1, 1, 1) or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class VideoPhysicsDecoder(nn.Module):
    """
    Video -> Physics Parameters decoder.
    InternViT backbone (frozen) -> 3D ResNet -> Temporal Attention -> Per-point MLP heads.

    Predicts per-point: stiffness (k), mass (m), damping (damp)
    Predicts global: friction (fric_k)
    """

    def __init__(self, n_points=2048, k_neighbors=256,
                 backbone_name="OpenGVLab/InternViT-300M-448px-V2_5",
                 freeze_backbone=True, d_model=512, nhead=8,
                 num_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.n_points = n_points
        self.k_neighbors = k_neighbors

        self.backbone = AutoModel.from_pretrained(
            backbone_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        cfg = getattr(self.backbone, "config", None)
        self.bb_hidden = getattr(cfg, "hidden_size", 1024)
        self.bb_image_size = getattr(cfg, "image_size", 448)

        self.token_proj = nn.Linear(self.bb_hidden, d_model)
        self.query_embed = nn.Parameter(torch.randn(n_points, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        mlp_hidden = d_model
        self.head_k = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1),
        )
        self.head_m = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1),
        )
        self.head_damp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, 1),
        )
        self.head_fric = nn.Sequential(
            nn.Linear(d_model, mlp_hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden // 2, 1),
        )

        self.register_buffer("bb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("bb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _extract_tokens(self, frames):
        """[B*T, 3, H, W] -> [B*T, Np, C]"""
        x = F.interpolate(frames, size=(self.bb_image_size, self.bb_image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.bb_mean) / self.bb_std
        x = x.to(dtype=torch.bfloat16)
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state[:, 1:, :]
        return tokens

    @staticmethod
    def _pos_enc_2d(h, w, d_model, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        y, x = y.reshape(-1).float(), x.reshape(-1).float()
        dim = d_model // 2
        div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe_x = torch.zeros(x.numel(), dim, device=device)
        pe_y = torch.zeros(y.numel(), dim, device=device)
        pe_x[:, 0::2] = torch.sin(x[:, None] * div)
        pe_x[:, 1::2] = torch.cos(x[:, None] * div)
        pe_y[:, 0::2] = torch.sin(y[:, None] * div)
        pe_y[:, 1::2] = torch.cos(y[:, None] * div)
        pe = torch.cat([pe_x, pe_y], dim=-1)
        if pe.size(-1) < d_model:
            pe = F.pad(pe, (0, d_model - pe.size(-1)))
        return pe

    @staticmethod
    def _pos_enc_1d(T, d_model, device):
        t = torch.arange(T, device=device).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(T, d_model, device=device)
        pe[:, 0::2] = torch.sin(t[:, None] * div)
        pe[:, 1::2] = torch.cos(t[:, None] * div)
        return pe

    def forward(self, video):
        """
        Args:
            video: [B, 3, T, H, W] in [0, 1]
        Returns:
            dict with k [B, N], m [B, N], damp [B, N], fric_k [B]
        """
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        tokens_bt = self._extract_tokens(frames).to(torch.float32)
        Np = tokens_bt.size(1)
        Hp = Wp = int(Np ** 0.5)

        tokens = tokens_bt.view(B, T, Np, -1)
        d = self.token_proj.out_features

        pos_hw = self._pos_enc_2d(Hp, Wp, d, tokens.device).unsqueeze(0).unsqueeze(0)
        pos_t = self._pos_enc_1d(T, d, tokens.device).unsqueeze(0).unsqueeze(2)

        tokens = self.token_proj(tokens) + pos_hw + pos_t
        memory = tokens.reshape(B, T * Np, -1)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        point_feat = self.decoder(tgt=queries, memory=memory)

        k = 100.0 + 1000.0 * torch.sigmoid(self.head_k(point_feat).squeeze(-1))
        m = 0.2 + 5.8 * torch.sigmoid(self.head_m(point_feat).squeeze(-1))
        damp = 0.1 + 4.9 * torch.sigmoid(self.head_damp(point_feat).squeeze(-1))
        fric_k = torch.sigmoid(self.head_fric(point_feat).squeeze(-1).mean(dim=1))

        return {'k': k, 'm': m, 'damp': damp, 'fric_k': fric_k}
