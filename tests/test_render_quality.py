"""Compare canonical vs render with different color conversions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import imageio
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from plyfile import PlyData

from models.renderer import GaussianRenderer

device = torch.device('cuda:0')
plant_dir = Path('data/plants_3dgs/a_single_adenium_with_a_bulbous_base_and_twisted_branches_wi_s42')

plydata = PlyData.read(str(plant_dir / 'gaussian.ply'))
v = plydata['vertex']
xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32, device=device)
scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32, device=device))
rots = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1), dtype=torch.float32, device=device)
rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-8)
opacities = torch.sigmoid(torch.tensor(v['opacity'][:, None], dtype=torch.float32, device=device))
sh_dc = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32, device=device)

print(f"f_dc range: [{sh_dc.min():.3f}, {sh_dc.max():.3f}]")

# Try different conversions
SH_C0 = 0.28209479177387814
colors_sh = (SH_C0 * sh_dc + 0.5).clamp(0.0, 1.0)
colors_sigmoid = torch.sigmoid(sh_dc)
print(f"SH conversion range: [{colors_sh.min():.3f}, {colors_sh.max():.3f}]")
print(f"Sigmoid conversion range: [{colors_sigmoid.min():.3f}, {colors_sigmoid.max():.3f}]")

renderer = GaussianRenderer(
    image_height=512, image_width=512, fov=40,
    bg_color=[0.0, 0.0, 0.0], sh_degree=0,
).to(device)
camera = renderer.get_camera(azimuth=0, elevation=14, radius=2.0, target=xyz.mean(0).detach())

for name, colors in [('sh', colors_sh), ('sigmoid', colors_sigmoid)]:
    with torch.no_grad():
        img = renderer.render_frame(xyz, scales, rots, opacities, colors, camera)
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.imwrite(f'outputs/render_{name}.png', img_np)
    print(f"Saved outputs/render_{name}.png")

# Also pass shs directly (this should match TRELLIS convention)
shs = sh_dc.unsqueeze(1)  # [N, 1, 3]
with torch.no_grad():
    img = renderer.render_frame(xyz, scales, rots, opacities, None, camera, shs=shs)
img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
imageio.imwrite('outputs/render_shs_direct.png', img_np)
print("Saved outputs/render_shs_direct.png")
