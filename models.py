# ===========================================================
#  Joint discrete + continuous diffusion for precipitation
# ===========================================================
import math, random, itertools
import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# -----------------------------------------------------------
#  Simple UNet backbone (works for both tasks)
# -----------------------------------------------------------
class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc  = nn.Sequential(nn.Conv2d(in_ch,  base, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU())
        self.up   = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU())
        self.out  = nn.Conv2d(base, out_ch, 1)
    def forward(self, x, t_emb):
        """t_emb is a (B, *) scalar/embedding broadcast via add."""
        h1 = self.inc(x + t_emb)
        h2 = self.down(h1 + t_emb)
        h3 = self.up(h2 + t_emb)
        return self.out(h1 + h3)

# -----------------------------------------------------------
#  Larger UNet with two downsample/upsample stages
# -----------------------------------------------------------
class LargeUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(),
                nn.Conv2d(oc, oc, 3, padding=1), nn.ReLU(),
            )

        self.inc = block(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), block(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), block(base * 2, base * 4))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 4, base * 2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 2, base),
        )
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, t_emb):
        h1 = self.inc(x + t_emb)
        h2 = self.down1(h1 + t_emb)
        h3 = self.down2(h2 + t_emb)
        u1 = self.up1(h3 + t_emb) + h2
        u2 = self.up2(u1 + t_emb)
        return self.out(u2 + h1)

# -----------------------------------------------------------
#  Helper: sinusoidal timestep embedding  (1×H×W broadcast)
# -----------------------------------------------------------
def time_embedding(t, H, W, device):
    # t: (B,) in [0,T)
    half = 32
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=device) / half)
    emb = torch.cat([torch.sin(t[:,None]*freqs), torch.cos(t[:,None]*freqs)], dim=1)
    return emb[:,:,None,None].repeat(1,1,H,W)    # (B,64,H,W)

# -----------------------------------------------------------
#  1) Discrete diffusion for wet/dry mask  (Austin et al. 2021)
# -----------------------------------------------------------
class BinaryD3PM:
    """Two-class diffusion where q(x_t=orig)=1-β̄_t, else uniform(0/1)."""
    def __init__(self, T=1000, beta0=1e-4, betaT=0.02, device='cpu'):
        self.T = T
        beta = torch.linspace(beta0, betaT, T+1, device=device)   # 0…T inclusive
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)             # (T+1,)
        self.device = device
    # ---- forward -------------------------------------------------
    def q_sample(self, x0, t):
        """x0 (B,1,H,W) ∈{0,1}; t (B,)"""
        B, _, H, W = x0.shape
        keep_prob = self.alpha_bar[t]          # (B,)
        keep_mask = torch.bernoulli(keep_prob.view(B,1,1,1).expand_as(x0))
        rand_bits = torch.bernoulli(torch.full_like(x0, .5))
        return torch.where(keep_mask.bool(), x0, rand_bits).long()
    # ---- objective / loss ----------------------------------------
    def loss(self, model_out, x0, t):
        """
        model_out: logits (B,2,H,W) for p_theta(x0 | x_t,t)
        x0      : (B,1,H,W) ground-truth originals
        """
        x0_long = x0.squeeze(1).long()         # BCEWithLogits needs class idx
        ce = F.cross_entropy(model_out, x0_long, reduction='none')      # (B,H,W)
        return ce.mean()

# -----------------------------------------------------------
#  2) Continuous DDPM for rainfall intensity
# -----------------------------------------------------------
class GaussianDDPM:
    def __init__(self, T=1000, beta0=1e-4, betaT=0.02, device='cpu'):
        self.T = T
        beta = torch.linspace(beta0, betaT, T+1, device=device)
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.device = device
    # ---- forward -------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bar[t].sqrt().view(-1,1,1,1)
        sqrt_om = (1-self.alpha_bar[t]).sqrt().view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_om * noise, noise
    # ---- objective / loss ----------------------------------------
    def loss(self, model_out, noise, wet_mask):
        """MSE only on wet pixels"""
        mse = ((model_out - noise)**2) * wet_mask
        denom = max(wet_mask.sum(), 1)       # avoid /0
        return mse.sum() / denom

# -----------------------------------------------------------
#  3) Lightning-style wrapper for joint training
# -----------------------------------------------------------
class JointRainDiffuser(nn.Module):
    def __init__(self, T=1000, device='cpu'):
        super().__init__()
        self.device = device
        self.disc = BinaryD3PM(T=T, device=device)
        self.cont = GaussianDDPM(T=T, device=device)
        # TWO separate UNets ------------------------------
        self.unet_mask = TinyUNet(in_ch=1+64, out_ch=2).to(device)  # logits
        self.unet_rain = TinyUNet(in_ch=2+64, out_ch=1).to(device)  # ε-hat
    def forward(self, batch):
        # The DataLoader returns a tensor of shape (B, 1, H, W) directly,
        # not a dictionary.
        x_amt = batch.to(self.device)    # (B,1,H,W)
        x0_mask = (x_amt > 0).float()              # binary wet/dry
        B,_,H,W = x_amt.shape
        # ---------- 1: discrete branch -----------------
        t_d = torch.randint(1, self.disc.T+1, (B,), device=self.device)
        x_t_mask = self.disc.q_sample(x0_mask, t_d)
        t_emb = time_embedding(t_d, H, W, self.device)
        disc_in = torch.cat([x_t_mask.float(), t_emb], dim=1)
        logits = self.unet_mask(disc_in, 0)        # t already embedded
        loss_mask = self.disc.loss(logits, x0_mask, t_d)
        # ---------- 2: continuous branch ---------------
        t_c = torch.randint(1, self.cont.T+1, (B,), device=self.device)
        x_t_amt, eps = self.cont.q_sample(x_amt, t_c)
        cond = torch.cat([x_t_amt, x0_mask], dim=1)         # add mask as input
        t_emb2 = time_embedding(t_c, H, W, self.device)
        eps_hat = self.unet_rain(torch.cat([cond, t_emb2], 1), 0)
        loss_amt = self.cont.loss(eps_hat, eps, x0_mask)    # only wet pixels
        return loss_mask, loss_amt

# -----------------------------------------------------------
#  4) Training loop (plain PyTorch) --------------------------
# -----------------------------------------------------------
def train(model, loader, epochs=10, lr=2e-4, lam=1.0, device='cuda'):
    opt = torch.optim.AdamW(itertools.chain(model.unet_mask.parameters(),
                                            model.unet_rain.parameters()), lr=lr)
    for ep in range(epochs):
        tot_mask = tot_amt = 0.0
        for idx, batch in enumerate(loader):
            opt.zero_grad()
            # Pass the batch tensor directly to the model
            Lmask, Lamt = model(batch)
            loss = Lmask + lam * Lamt
            # if idx % 10 == 0:
            #   print(loss.item())
            loss.backward()
            opt.step()
            tot_mask += Lmask.item();  tot_amt += Lamt.item()
        print(f"Epoch {ep+1:02d}: Lmask={tot_mask/len(loader):.4f} | "
              f"Lamt={tot_amt/len(loader):.4f}")