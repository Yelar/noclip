"""Debug: write all textures as one sheet. python -m sim._texsheet OUT.jpg"""
import sys, time, torch
from PIL import Image
from . import tex
t0 = time.time()
a = tex.build("cuda", keep=True)
print("build", round(time.time() - t0, 2), "s", a["atlas"].shape, len(a["names"]))
T = a["tex"]; n = len(T); cols = 12; rows = (n + cols - 1) // cols; s = 128
sheet = torch.zeros(3, rows * s, cols * s)
for i, t in enumerate(T):
    im = torch.nn.functional.avg_pool2d(t[None], 4)[0].cpu()
    rgb = im[:3] if i < a["n_surf"] else (im[:3] ** 2 / im[3:].clamp(min=1e-3)).sqrt() * im[3:] + (1 - im[3:]) * torch.tensor([0.3, 0.1, 0.3]).view(3, 1, 1)
    sheet[:, (i // cols) * s:(i // cols + 1) * s, (i % cols) * s:(i % cols + 1) * s] = rgb
Image.fromarray((sheet.permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()).save(sys.argv[1], quality=88)
print(list(a["names"]))
