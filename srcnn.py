#Import necessary libraries.
import os, glob, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

#Configuration

HR_DIR = "data/sample_images"      # put a few JPG/PNG here (DIV2K later)
SCALE = 2                      # 2x to start; try 4x later
PATCH = 96                     # HR patch size
BATCH = 8
EPOCHS = 5
LRATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


#

def bicubic_down_up(hr_img, scale=SCALE):
    w, h = hr_img.size
    lr = hr_img.resize((w//scale, h//scale), Image.BICUBIC)
    lr_up = lr.resize((w, h), Image.BICUBIC)  # classic SRCNN input
    return lr_up, hr_img

def to_tensor(img):  # [0,1] CHW float
    return torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], 3).numpy()/255.0)
    ).permute(2,0,1).float()

class HRFolder(Dataset):
    def __init__(self, root, patch=PATCH):
        self.paths = sorted(sum([glob.glob(str(Path(root)/ext)) for ext in ("*.png","*.jpg","*.jpeg","*.bmp")], []))
        self.patch = patch
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # random crop on HR, then make LR_up to same size
        W,H = img.size
        if W<self.patch or H<self.patch:
            img = img.resize((max(W,self.patch), max(H,self.patch)), Image.BICUBIC)
            W,H = img.size
        x = random.randint(0, W-self.patch); y = random.randint(0, H-self.patch)
        hr = img.crop((x,y,x+self.patch,y+self.patch))
        lr_up, hr = bicubic_down_up(hr)
        return to_tensor(lr_up), to_tensor(hr)

def psnr(pred, target, maxv=1.0):
    mse = F.mse_loss(pred, target)
    return 20*math.log10(maxv) - 10*math.log10(mse.item()+1e-12)

# ---- models ----
class SRCNN(nn.Module):
    # Paper-ish kernel sizes: 9-1-5 (we’ll use 9-5-5; feel free to tweak)
    def __init__(self, c=3):
        super().__init__()
        self.conv1 = nn.Conv2d(c, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
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
    best = -1

    for ep in range(1, EPOCHS+1):
        net.train()
        pbar = tqdm(dl, desc=f"Epoch {ep}")
        for lr_up, hr in pbar:
            lr_up, hr = lr_up.to(DEVICE), hr.to(DEVICE)
            sr = net(lr_up)
            loss = F.mse_loss(sr, hr)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
        # quick val on a few samples
        net.eval()
        with torch.no_grad():
            lr_up, hr = ds[0]
            lr_up = lr_up.unsqueeze(0).to(DEVICE); hr = hr.unsqueeze(0).to(DEVICE)
            sr = net(lr_up).clamp(0,1)
            bic_psnr = psnr(lr_up, hr);  sr_psnr = psnr(sr, hr)
            print(f"[Val] Bicubic PSNR: {bic_psnr:.2f} dB | SRCNN PSNR: {sr_psnr:.2f} dB")
            if sr_psnr > best:
                best = sr_psnr
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(net.state_dict(), "checkpoints/srcnn_best.pth")
    # save a visual
    with torch.no_grad():
        lr_up, hr = ds[0]
        lr_up = lr_up.unsqueeze(0).to(DEVICE); hr = hr.squeeze(0).to(DEVICE)
        sr = net(lr_up).squeeze(0).clamp(0,1).cpu()
        def to_pil(t): return Image.fromarray((t.permute(1,2,0).numpy()*255).astype('uint8'))
        to_pil(hr).save("hr_gt.png")
        to_pil(lr_up.squeeze(0).cpu()).save("lr_bicubic.png")
        to_pil(sr).save("sr_srcnn.png")
        print("Saved: hr_gt.png, lr_bicubic.png, sr_srcnn.png")
if __name__ == "__main__":
    main()
