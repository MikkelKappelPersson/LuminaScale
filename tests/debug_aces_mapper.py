import torch
from luminascale.models.aces_mapper import ACESMapper

def test_mapper_shape():
    batch_size = 2
    height, width = 128, 128
    
    # Initialize mapper
    # Smaller params for quick test
    model = ACESMapper(
        num_luts=2,
        lut_dim=17,
        num_lap=2,
        num_residual_blocks=1,
        sft_embed_dim=16, # Divisible by 8 (max num_heads in depth 3-4 are lower)
        sft_num_heads=[1, 2, 4, 8, 8, 4, 2, 1]
    )
    
    # Dummy input [B, 3, H, W] in [0, 1]
    dummy_input = torch.rand(batch_size, 3, height, width)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    assert output.shape == dummy_input.shape, f"Shape mismatch: {output.shape} != {dummy_input.shape}"
    print("Success: Output shape matches input shape.")

if __name__ == "__main__":
    try:
        test_mapper_shape()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
