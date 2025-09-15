#Import necessary libraries.
import os, glob, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np

#Configuration

HR_DIR = "data/sample_images"      # put a few JPG/PNG here (DIV2K later)
SCALE = 2                      # 2x to start; try 4x later
PATCH = 64                     # HR patch size
BATCH = 4
EPOCHS = 40
LRATE = 5e-4
REPEATS = 50
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


#

def bicubic_down_up(hr_img, scale=SCALE):
    w, h = hr_img.size
    lr = hr_img.resize((w//scale, h//scale), Image.BICUBIC)
    lr_up = lr.resize((w, h), Image.BICUBIC)  # classic SRCNN input
    return lr_up, hr_img

def to_tensor(img: Image.Image):
    """PIL -> torch tensor, CHW in [0,1], float32."""
    arr = np.array(img, dtype=np.float32) / 255.0     # HWC
    if arr.ndim == 2:  # grayscale -> add channel
        arr = arr[:, :, None]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

class HRFolder(Dataset):
    def __init__(self, root, patch=PATCH, repeats=REPEATS):
        self.paths = sorted([str(p) for ext in ("*.png","*.jpg","*.jpeg","*.bmp")
                             for p in Path(root).glob(ext)])
        self.patch = patch
        self.repeats = repeats
    def __len__(self): return len(self.paths) * self.repeats
    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        img = Image.open(path).convert("RGB")
        W, H = img.size
        if W < self.patch or H < self.patch:
            img = img.resize((max(W, self.patch), max(H, self.patch)), Image.BICUBIC)
            W, H = img.size
        x = random.randint(0, W - self.patch); y = random.randint(0, H - self.patch)
        hr = img.crop((x, y, x + self.patch, y + self.patch))
        lr = hr.resize((self.patch // SCALE, self.patch // SCALE), Image.BICUBIC)
        lr_up = lr.resize((self.patch, self.patch), Image.BICUBIC)
        return to_tensor(lr_up), to_tensor(hr)


def psnr(pred: torch.Tensor, target: torch.Tensor, maxv: float = 1.0) -> float:
    # pred/target: NCHW or CHW in [0,1]
    if pred.dim() == 3: pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    mse = F.mse_loss(pred, target)
    return 20 * math.log10(maxv) - 10 * math.log10(mse.item() + 1e-12)

# ---- models ----
class SRCNN(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)   # 1×1 per paper
        self.conv3 = nn.Conv2d(32, c, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x



def main():
    ds = HRFolder(HR_DIR)
    assert len(ds)>0, f"Put a few images in {HR_DIR}"
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    net = SRCNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LRATE)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    best = -1

    for ep in range(1, EPOCHS+1):
        net.train()
        pbar = tqdm(dl, desc=f"Epoch {ep}")
        for lr_up, hr in pbar:
            lr_up, hr = lr_up.to(DEVICE), hr.to(DEVICE)
            sr = net(lr_up)
            loss = F.l1_loss(sr, hr)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.detach()))

        # quick val on a few samples
        #net.eval()
        #with torch.no_grad():
        #    lr_up, hr = ds[0]
        #    lr_up = lr_up.unsqueeze(0).to(DEVICE); hr = hr.unsqueeze(0).to(DEVICE)
        #    sr = net(lr_up).clamp(0,1)
        #    bic_psnr = psnr(lr_up, hr);  sr_psnr = psnr(sr, hr)
        #   print(f"[Val] Bicubic PSNR: {bic_psnr:.2f} dB | SRCNN PSNR: {sr_psnr:.2f} dB")
        #    if sr_psnr > best:
        #        best = sr_psnr
        #        os.makedirs("checkpoints", exist_ok=True)
        #        torch.save(net.state_dict(), "checkpoints/srcnn_best.pth")

        # replace your current quick val block
        net.eval()
        with torch.no_grad():
            K = 8
            bic_sum = 0.0
            src_sum = 0.0
            for _ in range(K):
                lr_up_cpu, hr_cpu = ds[random.randrange(len(ds))]
                lr_up = lr_up_cpu.unsqueeze(0).to(DEVICE)
                hr    = hr_cpu.unsqueeze(0).to(DEVICE)
                sr    = net(lr_up).clamp(0, 1)

                bic_sum += psnr(lr_up, hr)
                src_sum += psnr(sr, hr)

            bic_psnr = bic_sum / K
            sr_psnr  = src_sum / K
            print(f"[Val] Bicubic PSNR: {bic_psnr:.2f} dB | SRCNN PSNR: {sr_psnr:.2f} dB")

            if sr_psnr > best:
                best = sr_psnr
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(net.state_dict(), "checkpoints/srcnn_best.pth")
        
        sched.step()






    # save a visual

    with torch.no_grad():
        lr_up_cpu, hr_cpu = ds[0]                   # CPU tensors, CHW
        assert lr_up_cpu.shape == hr_cpu.shape, f"Shape mismatch: {lr_up_cpu.shape} vs {hr_cpu.shape}"
        assert lr_up_cpu.min() >= 0 and lr_up_cpu.max() <= 1, "lr_up not in [0,1]"
        assert hr_cpu.min()    >= 0 and hr_cpu.max()    <= 1, "hr not in [0,1]"

        lr_up = lr_up_cpu.unsqueeze(0).to(DEVICE)  # 1CHW
        sr = net(lr_up).squeeze(0).clamp(0, 1)     # CHW

        to_pil(hr_cpu).save("hr_gt.png")
        to_pil(lr_up_cpu).save("lr_bicubic.png")
        to_pil(sr).save("sr_srcnn.png")
        print("Saved: hr_gt.png, lr_bicubic.png, sr_srcnn.png")




def to_pil(tensor: torch.Tensor) -> Image.Image:
    """torch tensor (C,H,W) in [0,1] on any device -> PIL Image (uint8)."""
    t = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()  # HWC
    t = (t * 255.0 + 0.5).astype(np.uint8)
    if t.shape[2] == 1:
        t = t[:, :, 0]
    return Image.fromarray(t)


if __name__ == "__main__":
    main()
