import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFrequencyMask(nn.Module):
    """
    Base class for 2D frequency masks.

    forward(radius_map) should return a tensor of shape (H, W):
        - either boolean (hard mask: inside vs outside)
        - or float in [0, 1] (soft mask: blending weights)
    """
    def forward(self, radius_map: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class RadialHardFrequencyMask(BaseFrequencyMask):
    """
    Binary radial low-pass mask: 1 inside cutoff, 0 outside.

    This reproduces the original FreqPure behavior when used in PurificationForward.
    You can optionally make `cutoff` learnable, but the mask itself is still hard.
    """

    def __init__(self, cutoff: float, device: torch.device, learnable: bool = False):
        super().__init__()
        cutoff_tensor = torch.as_tensor(float(cutoff), dtype=torch.float32, device=device)
        if learnable:
            # You *can* optimize this scalar, but gradients do not flow through
            # the hard threshold itself. For truly learnable masks, prefer RadialSoftFrequencyMask.
            self.cutoff = nn.Parameter(cutoff_tensor)
        else:
            self.register_buffer("cutoff", cutoff_tensor)

    def forward(self, radius_map: torch.Tensor) -> torch.Tensor:
        # radius_map: (H, W)
        return (radius_map <= self.cutoff).to(dtype=torch.bool)


class RadialSoftFrequencyMask(BaseFrequencyMask):
    """
    Smooth radial low-pass mask: values in (0, 1).

    mask(r) = sigmoid( sharpness * (cutoff - r) )

    - For r << cutoff: mask ≈ 1  (keep mostly from adversarial input)
    - For r >> cutoff: mask ≈ 0  (keep mostly from diffusion estimate)

    Both `cutoff` and `sharpness` are parameterized in log-space for
    numerical stability and positive constraints.
    """

    def __init__(
        self,
        init_cutoff: float,
        init_sharpness: float = 10.0,
        device: torch.device | None = None,
        learnable: bool = True,
    ):
        super().__init__()
        device = device if device is not None else torch.device("cpu")

        cutoff = torch.as_tensor(float(init_cutoff), dtype=torch.float32, device=device)
        sharpness = torch.as_tensor(float(init_sharpness), dtype=torch.float32, device=device)

        if learnable:
            self.log_cutoff = nn.Parameter(cutoff.log())
            self.log_sharpness = nn.Parameter(sharpness.log())
        else:
            self.register_buffer("log_cutoff", cutoff.log())
            self.register_buffer("log_sharpness", sharpness.log())

    @property
    def cutoff(self) -> torch.Tensor:
        return torch.exp(self.log_cutoff)

    @property
    def sharpness(self) -> torch.Tensor:
        return torch.exp(self.log_sharpness)

    def forward(self, radius_map: torch.Tensor) -> torch.Tensor:
        # Ensure same device / dtype
        radius_map = radius_map.to(self.cutoff.device).float()
        c = self.cutoff
        s = self.sharpness
        # Smooth approximation of a radial indicator
        return torch.sigmoid(s * (c - radius_map))