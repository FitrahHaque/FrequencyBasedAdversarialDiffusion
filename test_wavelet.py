"""Quick test to verify wavelet reconstruction."""
import torch
from wavelet_utils import multi_level_dwt, multi_level_idwt

def test_wavelet_reconstruction():
    print("Testing Wavelet Reconstruction...")
    
    # Test with CIFAR-10 size
    x = torch.rand(2, 3, 32, 32)
    
    for levels in [1, 2, 3]:
        LL, coeffs, pads = multi_level_dwt(x, levels=levels)
        x_rec = multi_level_idwt(LL, coeffs, pads)
        
        diff = (x - x_rec).abs().max().item()
        print(f"Level {levels}: Max reconstruction error = {diff:.2e}")
        assert diff < 1e-5, f"Reconstruction error too high for level {levels}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_wavelet_reconstruction()
