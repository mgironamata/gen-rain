# ================================================================
#  Synthetic precipitation generator based on a 2-D Gaussian Field
# ================================================================
import math
import torch
from torch.utils.data import Dataset

# ---------- helper: RBF / Matérn kernel -------------------------
def rbf_kernel(X, length_scale: float, variance: float = 1.0):
    """Squared-exponential kernel K_ij = σ² exp(-‖xi-xj‖² / 2ℓ²)."""
    sqdist = torch.cdist(X, X, p=2.0) ** 2
    return variance * torch.exp(-0.5 * sqdist / length_scale**2)

# ---------- core class ------------------------------------------
class GaussianRainfieldGenerator:
    """
    Quickly samples spatial rainfall fields on a regular grid.
      • latent GP (zero-mean, RBF kernel)  -> probability of rain
      • logistic transform + threshold    -> wet/dry mask
      • positive rain amounts (Gamma/LogNormal) on wet pixels
    """
    def __init__(
        self,
        grid_height      = 64,
        grid_width       = 64,
        length_scale     = 0.15,    # in units of domain ∈[0,1]
        gp_variance      = 1.0,
        wet_threshold    = 0.5,     # logits threshold (~50 % coverage)
        amount_dist      = "gamma", # or "lognormal"
        gamma_shape      = 2.0,
        gamma_scale      = 5.0,     # mm
        lognorm_mu       = 1.0,
        lognorm_sigma    = 0.5,
        correlated_intensity = False,
        amount_length_scale = None,
        amount_variance = 1.0,
        device           = "cpu"
    ):
        self.H, self.W  = grid_height, grid_width
        self.amount_dist = amount_dist.lower()
        self.wet_thr     = wet_threshold
        self.device      = device

        # Assign the distribution parameters to self
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma

        self.corr_intensity = correlated_intensity
        if amount_length_scale is None:
            amount_length_scale = length_scale
        self.amount_length_scale = amount_length_scale
        self.amount_variance = amount_variance


        # create (x,y) coordinates in [0,1]×[0,1]
        xs, ys = torch.linspace(0, 1, grid_width), torch.linspace(0, 1, grid_height)
        X, Y   = torch.meshgrid(xs, ys, indexing="xy")
        coords = torch.stack((X.flatten(), Y.flatten()), dim=1).to(device)

        # pre-compute Cholesky factor of the covariance matrix for occurrence
        K   = rbf_kernel(coords, length_scale, gp_variance)        # (N,N)
        jitter = 1e-3 * torch.eye(K.shape[0], device=device)       # numerical stability
        self.L = torch.linalg.cholesky(K + jitter)                 # lower-triangular
        self.N = K.shape[0]

        # Optional second GP for rainfall amounts
        if self.corr_intensity:
            K_amt = rbf_kernel(coords, self.amount_length_scale, self.amount_variance)
            jitter_amt = 1e-3 * torch.eye(K_amt.shape[0], device=device)
            self.L_amt = torch.linalg.cholesky(K_amt + jitter_amt)
        else:
            self.L_amt = None

    @torch.no_grad()
    def _sample_latent_gp(self, n_samples: int, L=None) -> torch.Tensor:
        """Draw n latent Gaussian fields ∼ GP(0, K); returns (n, H, W)."""
        if L is None:
            L = self.L
        N = L.shape[0]
        z = torch.randn(n_samples, N, device=self.device)
        f = (L @ z.T).T                      # (n, N)
        return f.view(n_samples, self.H, self.W)  # reshape into images

    @torch.no_grad()
    def sample_precip(self, n_samples: int = 1, occurrence_only: bool = False) -> torch.Tensor:
        """
        Generate rainfall fields.

        Parameters
        ----------
        n_samples : int
            Number of fields to generate.
        occurrence_only : bool, optional
            If ``True`` return only the binary wet/dry mask.  If ``False``
            return precipitation amounts in millimetres.  Defaults to ``False``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_samples, 1, H, W)`` representing precipitation
            occurrence or amount.
        """
        latent = self._sample_latent_gp(n_samples)

        # ---------- Stage 1: wet/dry mask -----------------------
        prob_wet = torch.sigmoid(latent)          # logistic GP(0, K)
        mask      = (prob_wet > self.wet_thr).float()

        if occurrence_only:
            return mask.unsqueeze(1)

        # ---------- Stage 2: intensities on wet pixels ----------
        if self.corr_intensity:
            latent_amt = self._sample_latent_gp(n_samples, self.L_amt)
            mu, sigma = self.lognorm_mu, self.lognorm_sigma
            amounts = torch.exp(mu + sigma * latent_amt)
        elif self.amount_dist == "gamma":
            shape, scale = self.gamma_shape, self.gamma_scale
            amounts = torch.distributions.Gamma(shape, 1/scale).sample(latent.shape)
        else:   # log-normal with independent pixels
            mu, sigma = self.lognorm_mu, self.lognorm_sigma
            amounts = torch.distributions.LogNormal(mu, sigma).sample(latent.shape)

        rainfall = mask * amounts                 # element-wise
        return rainfall.unsqueeze(1)              # add channel dim

    # ---------- optional convenience: PyTorch Dataset ----------
    def make_dataset(self, n_samples: int, occurrence_only: bool = False):
        """Return a simple :class:`torch.utils.data.Dataset` of generated fields."""
        gen = self
        class _Rainset(Dataset):
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                # Fetch a single sample according to the requested output type
                return gen.sample_precip(1, occurrence_only=occurrence_only)[0]

        return _Rainset()
