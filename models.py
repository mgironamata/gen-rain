# ===========================================================
#  Joint discrete + continuous diffusion for precipitation
# ===========================================================
import math, random, itertools
import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Use GPU if available during sampling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Encoder
        self.inc = block(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), block(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), block(base * 2, base * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), block(base * 4, base * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), block(base * 8, base * 16))

        # Bottleneck
        self.bottleneck = block(base * 16, base * 16)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 16, base * 8),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 8, base * 4),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 4, base * 2),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            block(base * 2, base),
        )
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, t_emb):
        h1 = self.inc(x + t_emb)
        h2 = self.down1(h1 + t_emb)
        h3 = self.down2(h2 + t_emb)
        h4 = self.down3(h3 + t_emb)
        h5 = self.down4(h4 + t_emb)
        b = self.bottleneck(h5 + t_emb)
        u1 = self.up1(b + t_emb) + h4
        u2 = self.up2(u1 + t_emb) + h3
        u3 = self.up3(u2 + t_emb) + h2
        u4 = self.up4(u3 + t_emb)
        return self.out(u4 + h1)

# -----------------------------------------------------------
#  Transformer backbone for image-like data
# -----------------------------------------------------------
class SimpleVisionTransformer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, img_size=32, patch_size=4, dim=128, depth=4, heads=4, mlp_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_ch * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
        )
        self.out_ch = out_ch
        self.img_size = img_size

    def forward(self, x, t_emb):
        # x: (B, C, H, W), t_emb: (B, 1, 1, 1)
        B, C, H, W = x.shape
        p = self.patch_size
        # Add t_emb to x
        x = x + t_emb
        # Unfold into patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, nH, nW, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)  # (B, num_patches, patch_dim)
        # Patch embedding
        tokens = self.patch_embed(patches)  # (B, num_patches, dim)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        # Optionally, prepend cls token (not used here)
        # Transformer
        tokens = self.transformer(tokens)
        # Project back to patches
        out = self.head(tokens)  # (B, num_patches, patch_dim)
        # Fold back to image
        out = out.view(B, H // p, W // p, C, p, p).permute(0, 3, 1, 4, 2, 5)
        out = out.contiguous().view(B, C, H, W)
        # Final conv to get desired output channels
        conv_out = nn.Conv2d(C, self.out_ch, 1).to(x.device)
        return conv_out(out)
    
class LargeVisionTransformer(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, img_size=32, patch_size=4, dim=512, depth=16, heads=16, mlp_dim=1024):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_ch * patch_size * patch_size

        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
        )
        self.out_ch = out_ch
        self.img_size = img_size

        self.final_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x + t_emb
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)
        tokens = self.patch_embed(patches)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        tokens = self.transformer(tokens)
        out = self.head(tokens)
        out = out.view(B, H // p, W // p, C, p, p).permute(0, 3, 1, 4, 2, 5)
        out = out.contiguous().view(B, C, H, W)
        return self.final_conv(out)

# -----------------------------------------------------------
#  Helper: sinusoidal timestep embedding  (broadcastable scalar)
# -----------------------------------------------------------
def time_embedding(t, H, W, device):
    # t: (B,) in [0,T)
    half = 32
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=device) / half)
    emb = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=1)
    emb = emb.mean(dim=1, keepdim=True)                  # (B,1)
    return emb[:, :, None, None]                         # broadcastable scalar

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
    def __init__(self, T=1000, device='cpu', backbone='tiny_unet'):
        super().__init__()
        self.device = device
        self.disc = BinaryD3PM(T=T, device=device)
        self.cont = GaussianDDPM(T=T, device=device)
        self.backbone = backbone 
        
        # TWO separate UNets ------------------------------
        if self.backbone == 'tiny_unet':
            self.unet_mask = TinyUNet(in_ch=1, out_ch=2).to(device)  # logits
            self.unet_rain = TinyUNet(in_ch=2, out_ch=1).to(device)  # ε-hat
        elif self.backbone == 'large_unet':
            self.unet_mask = LargeUNet(in_ch=1, out_ch=2).to(device)
            self.unet_rain = LargeUNet(in_ch=2, out_ch=1).to(device)  # ε-hat
        elif self.backbone == 'tiny_vit':
            self.unet_mask = SimpleVisionTransformer(in_ch=1, out_ch=2, img_size=32).to(device)
            self.unet_rain = SimpleVisionTransformer(in_ch=2, out_ch=1, img_size=32).to(device)
        elif self.backbone == 'large_vit':
            self.unet_mask = LargeVisionTransformer(in_ch=1, out_ch=2, img_size=32).to(device)
            self.unet_rain = LargeVisionTransformer(in_ch=2, out_ch=1, img_size=32).to(device)

    def forward(self, batch, mask_only: bool = False):
        # The DataLoader returns a tensor of shape (B, 1, H, W) directly,
        # not a dictionary.
        x_amt = batch.to(self.device)    # (B,1,H,W)
        x0_mask = (x_amt > 0).float()              # binary wet/dry
        B,_,H,W = x_amt.shape
        # ---------- 1: discrete branch -----------------
        t_d = torch.randint(1, self.disc.T + 1, (B,), device=self.device)
        x_t_mask = self.disc.q_sample(x0_mask, t_d)
        t_emb = time_embedding(t_d, H, W, self.device)
        logits = self.unet_mask(x_t_mask.float(), t_emb)
        loss_mask = self.disc.loss(logits, x0_mask, t_d)
        if mask_only:
            loss_amt = torch.tensor(0.0, device=self.device)
        else:
            # ---------- 2: continuous branch ---------------
            t_c = torch.randint(1, self.cont.T + 1, (B,), device=self.device)
            x_t_amt, eps = self.cont.q_sample(x_amt, t_c)
            cond = torch.cat([x_t_amt, x0_mask], dim=1)  # add mask as input
            t_emb2 = time_embedding(t_c, H, W, self.device)
            eps_hat = self.unet_rain(cond, t_emb2)
            loss_amt = self.cont.loss(eps_hat, eps, x0_mask)    # only wet pixels
        return loss_mask, loss_amt

# -----------------------------------------------------------
#  4) Training loop (plain PyTorch) --------------------------
# -----------------------------------------------------------
def train(model, loader, epochs=10, lr=2e-4, lam=1.0, device='cuda', occurrence_only: bool = False):
    opt = torch.optim.AdamW(itertools.chain(model.unet_mask.parameters(),
                                            model.unet_rain.parameters()), lr=lr)
    for ep in range(epochs):
        tot_mask = tot_amt = 0.0
        for idx, batch in enumerate(loader):
            opt.zero_grad()
            # Pass the batch tensor directly to the model
            Lmask, Lamt = model(batch, mask_only=occurrence_only)
            loss = Lmask if occurrence_only else Lmask + lam * Lamt
            # if idx % 10 == 0:
            #   print(loss.item())
            loss.backward()
            opt.step()
            tot_mask += Lmask.item();  tot_amt += Lamt.item()
        if occurrence_only:
            print(f"Epoch {ep+1:02d}: Lmask={tot_mask/len(loader):.4f}")
        else:
            print(
                f"Epoch {ep+1:02d}: Lmask={tot_mask/len(loader):.4f} | "
                f"Lamt={tot_amt/len(loader):.4f}"
            )
        
# ---- sampling ------------------------------------------
@torch.no_grad()
def sample(model, n=4, occurrence_only: bool = False, size=(32, 32)):
    
    H = size[0] if isinstance(size, tuple) else size
    W = size[1] if isinstance(size, tuple) else size

    # --- 1) sample mask via reverse discrete process ----
    x_t = torch.randint(0,2,(n,1,H,W), device=device)    # start from noise
    for t in reversed(range(1,model.disc.T+1)):
        t_batch = torch.full((n,), t, device=device)
        t_emb = time_embedding(t_batch, H, W, device)
        logits = model.unet_mask(x_t.float(), t_emb)
        probs  = F.softmax(logits, 1)[:,1:2]             # prob(wet)
        x0_pred = (probs > 0.5).float()
        # simplistic posterior sample: copy pred
        x_t = x0_pred
    mask = x_t                                           # (n,1,H,W)
    if occurrence_only:
        return mask.cpu()
    # --- 2) sample intensity conditioned on mask --------
    y_t = torch.randn_like(mask)                         # Gaussian noise
    for t in reversed(range(1,model.cont.T+1)):
        t_b = torch.full((n,), t, device=device)
        t_emb = time_embedding(t_b, H, W, device)
        cond = torch.cat([y_t, mask], 1)
        eps_hat = model.unet_rain(cond, t_emb)
        alpha_bar_t = model.cont.alpha_bar[t_b].view(-1,1,1,1)
        y0_pred = (y_t - (1 - alpha_bar_t).sqrt() * eps_hat) / alpha_bar_t.sqrt()
        if t > 1:
            alpha_bar_prev = model.cont.alpha_bar[t_b - 1].view(-1,1,1,1)
            noise = torch.randn_like(y_t)
            y_t = alpha_bar_prev.sqrt() * y0_pred + (1 - alpha_bar_prev).sqrt() * noise
        else:
            y_t = y0_pred
    return (y_t * mask).cpu()             # final precip field
