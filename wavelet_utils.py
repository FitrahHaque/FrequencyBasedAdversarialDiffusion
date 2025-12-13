"""
Wavelet Utilities for FreqPure
Implements differentiable 2D Haar Wavelet Transform (DWT) and Inverse DWT (IDWT).
"""
import torch
import torch.nn.functional as F


def dwt_2d(x):
    """
    Performs a single-level 2D Haar DWT.
    Args:
        x: Input tensor of shape (B, C, H, W)
    Returns:
        LL, LH, HL, HH: Subband tensors of shape (B, C, H/2, W/2)
    """
    # Row decomposition
    r_even = x[:, :, 0::2, :]
    r_odd = x[:, :, 1::2, :]
    
    L = (r_even + r_odd) / 2  # Low-pass (average)
    H = (r_even - r_odd) / 2  # High-pass (difference)
    
    # Column decomposition
    L_even = L[:, :, :, 0::2]
    L_odd = L[:, :, :, 1::2]
    H_even = H[:, :, :, 0::2]
    H_odd = H[:, :, :, 1::2]
    
    LL = (L_even + L_odd) / 2  # Approximation (low-low)
    HL = (L_even - L_odd) / 2  # Vertical details (low-high)
    LH = (H_even + H_odd) / 2  # Horizontal details (high-low)
    HH = (H_even - H_odd) / 2  # Diagonal details (high-high)
    
    return LL, LH, HL, HH


def idwt_2d(LL, LH, HL, HH):
    """
    Performs a single-level 2D Inverse Haar DWT.
    Args:
        LL, LH, HL, HH: Subband tensors
    Returns:
        Reconstructed tensor of shape (B, C, H*2, W*2)
    """
    # Reconstruct columns
    L_even = LL + HL
    L_odd = LL - HL
    H_even = LH + HH
    H_odd = LH - HH
    
    B, C, H, W = LL.shape
    
    # Interleave columns
    L = torch.zeros(B, C, H, W * 2, device=LL.device, dtype=LL.dtype)
    L[:, :, :, 0::2] = L_even
    L[:, :, :, 1::2] = L_odd
    
    H_temp = torch.zeros(B, C, H, W * 2, device=LL.device, dtype=LL.dtype)
    H_temp[:, :, :, 0::2] = H_even
    H_temp[:, :, :, 1::2] = H_odd
    
    # Reconstruct rows
    r_even = L + H_temp
    r_odd = L - H_temp
    
    # Interleave rows
    x = torch.zeros(B, C, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
    x[:, :, 0::2, :] = r_even
    x[:, :, 1::2, :] = r_odd
    
    return x


def pad_if_needed(x):
    """Pads input to be divisible by 2."""
    h, w = x.shape[-2], x.shape[-1]
    pad_h = 1 if h % 2 != 0 else 0
    pad_w = 1 if w % 2 != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
    return x, (pad_h, pad_w)


def multi_level_dwt(x, levels=2):
    """
    Multi-level 2D DWT decomposition.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        levels: Number of decomposition levels
        
    Returns:
        LL_final: Coarsest approximation coefficients
        coeffs: List of (LH, HL, HH) tuples from fine to coarse
        pads_list: List of padding info for each level
    """
    coeffs = []
    pads_list = []
    
    curr_LL = x
    for i in range(levels):
        curr_LL, pads = pad_if_needed(curr_LL)
        pads_list.append(pads)
        
        LL, LH, HL, HH = dwt_2d(curr_LL)
        coeffs.append((LH, HL, HH))
        curr_LL = LL
        
    return curr_LL, coeffs, pads_list


def multi_level_idwt(LL, coeffs, pads_list):
    """
    Multi-level 2D Inverse DWT reconstruction.
    
    Args:
        LL: Coarsest approximation coefficients
        coeffs: List of (LH, HL, HH) tuples from fine to coarse
        pads_list: List of padding info for each level
        
    Returns:
        Reconstructed tensor
    """
    curr_LL = LL
    
    # Iterate backwards (coarse to fine)
    for i in range(len(coeffs) - 1, -1, -1):
        LH, HL, HH = coeffs[i]
        pad_h, pad_w = pads_list[i]
        
        curr_LL = idwt_2d(curr_LL, LH, HL, HH)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            h, w = curr_LL.shape[-2], curr_LL.shape[-1]
            curr_LL = curr_LL[:, :, :h - pad_h, :w - pad_w]
            
    return curr_LL
