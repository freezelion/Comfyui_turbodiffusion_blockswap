#!/usr/bin/env python3
"""
Simple test to verify BlockSwapWrapper compatibility with ComfyUI LoRA loading.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.block_swap_wrapper import create_block_swap_wrapper


class SimpleModel(nn.Module):
    """Simple model with model_config attribute."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
        # Add model_config like ComfyUI models
        class ModelConfig:
            def __init__(self):
                self.unet_config = type('obj', (object,), {
                    'in_channels': 4,
                    'out_channels': 4,
                    'model_channels': 320
                })()
        
        self.model_config = ModelConfig()
        self.model_type = "SD15"
    
    def forward(self, x):
        return self.linear(x)


def test_simple():
    """Simple test without recursion issues."""
    print("Testing simple BlockSwapWrapper compatibility...")
    
    # Create simple model
    model = SimpleModel()
    print(f"Original model has model_config: {hasattr(model, 'model_config')}")
    print(f"Original model.model_type: {model.model_type}")
    
    # Create wrapper
    device = torch.device("cpu")  # Use CPU to avoid CUDA issues
    try:
        wrapped = create_block_swap_wrapper(
            model=model,
            device=device,
            max_vram_gb=2.0,
            block_size_mb=50,
            swap_mode="adaptive",
            verbose=False
        )
        print("✓ Wrapper created successfully")
    except Exception as e:
        print(f"✗ Failed to create wrapper: {e}")
        return False
    
    # Test attribute access
    print(f"\nTesting attribute access:")
    print(f"  wrapped has model_config: {hasattr(wrapped, 'model_config')}")
    print(f"  wrapped has model_type: {hasattr(wrapped, 'model_type')}")
    
    if hasattr(wrapped, 'model_config'):
        print(f"  wrapped.model_config.unet_config.in_channels: {wrapped.model_config.unet_config.in_channels}")
        print("✓ model_config accessible")
    else:
        print("✗ model_config NOT accessible")
        return False
    
    # Test forward pass
    print(f"\nTesting forward pass:")
    try:
        x = torch.randn(2, 10)
        with torch.no_grad():
            y = wrapped(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("Simple test passed! BlockSwapWrapper supports ComfyUI attributes.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_simple()
    sys.exit(0 if success else 1)
