"""
Test script for Block Swap Wrapper functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a simple test model
class TestModel(nn.Module):
    def __init__(self, num_layers=8, hidden_size=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ])
        self.input_proj = nn.Linear(10, hidden_size)
        self.output_proj = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

def test_block_swap():
    """Test the block swap wrapper functionality."""
    print("Testing Block Swap Wrapper...")
    
    # Create test model
    model = TestModel(num_layers=8, hidden_size=512)
    
    # Estimate model size
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = total_params * 4 / (1024**2)  # Assuming float32
    print(f"Test model: {total_params:,} parameters, ~{total_size_mb:.1f} MB")
    
    # Test CPU mode
    print("\n1. Testing CPU mode...")
    try:
        from utils.block_swap_wrapper import create_block_swap_wrapper
        cpu_model = create_block_swap_wrapper(
            model=model,
            device=torch.device("cpu"),
            max_vram_gb=8.0,
            block_size_mb=50,
            swap_mode="adaptive",
            verbose=True
        )
        
        # Test forward pass
        test_input = torch.randn(2, 10)
        output = cpu_model(test_input)
        print(f"  CPU forward pass successful: output shape {output.shape}")
    except Exception as e:
        print(f"  CPU test failed: {e}")
    
    # Test CUDA mode if available
    if torch.cuda.is_available():
        print("\n2. Testing CUDA mode...")
        try:
            from utils.block_swap_wrapper import create_block_swap_wrapper
            
            # Test different swap modes
            swap_modes = ["adaptive", "layerwise", "chunkwise"]
            
            for mode in swap_modes:
                print(f"\n  Testing {mode} mode...")
                cuda_model = create_block_swap_wrapper(
                    model=model,
                    device=torch.device("cuda:0"),
                    max_vram_gb=2.0,  # Small limit for testing
                    block_size_mb=10,  # Small blocks
                    swap_mode=mode,
                    verbose=True
                )
                
                # Test forward pass
                test_input = torch.randn(2, 10).cuda()
                output = cuda_model(test_input)
                print(f"    Forward pass successful: output shape {output.shape}")
                
                # Get stats
                stats = cuda_model.get_stats()
                print(f"    Peak VRAM used: {stats.get('peak_vram_used', 0):.2f} GB")
                print(f"    Blocks processed: {stats.get('blocks_processed', 0)}")
                
                # Clean up
                del cuda_model
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"  CUDA test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n2. CUDA not available, skipping CUDA tests")
    
    print("\n3. Testing wrapper integration...")
    try:
        # Test that the wrapper can be imported and used
        from utils.block_swap_wrapper import BlockSwapWrapper, BlockSwapManager
        
        print("  BlockSwapWrapper import successful")
        print("  BlockSwapManager import successful")
        
        # Test manager functionality
        manager = BlockSwapManager(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            max_vram_gb=8.0,
            block_size_mb=100,
            swap_mode="adaptive",
            verbose=False
        )
        
        vram_info = manager.get_vram_info()
        print(f"  VRAM info: {vram_info}")
        
    except Exception as e:
        print(f"  Wrapper integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Block Swap Wrapper tests completed!")

if __name__ == "__main__":
    test_block_swap()
