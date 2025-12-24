#!/usr/bin/env python3
"""
Test script to verify BlockSwapWrapper compatibility with ComfyUI LoRA loading.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.block_swap_wrapper import create_block_swap_wrapper


class MockComfyUIModel(nn.Module):
    """
    Mock model that mimics ComfyUI's model interface with model_config attribute.
    """
    
    def __init__(self):
        super().__init__()
        # Create a simple model
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        ])
        
        # Add model_config attribute like ComfyUI models have
        self.model_config = type('obj', (object,), {
            'unet_config': type('obj', (object,), {
                'in_channels': 4,
                'out_channels': 4,
                'model_channels': 320,
                'attention_resolutions': [4, 2, 1],
                'num_res_blocks': 2,
                'channel_mult': [1, 2, 4, 4],
                'num_heads': 8,
                'use_spatial_transformer': True,
                'transformer_depth': 1,
                'context_dim': 768,
                'use_checkpoint': True,
                'legacy': False
            })()
        })()
        
        # Add other ComfyUI attributes
        self.model_type = "SD15"
        self.diffusion_model = self
        self.latent_format = "SD15"
        self.conditioning_key = "crossattn"
        self.parameterization = "eps"
        self.scale_factor = 0.18215
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_lora_compatibility():
    """Test that BlockSwapWrapper properly forwards ComfyUI attributes."""
    print("Testing BlockSwapWrapper compatibility with ComfyUI LoRA loading...")
    
    # Create mock ComfyUI model
    original_model = MockComfyUIModel()
    print(f"Original model has model_config: {hasattr(original_model, 'model_config')}")
    print(f"Original model.model_config.unet_config.in_channels: {original_model.model_config.unet_config.in_channels}")
    
    # Create block swap wrapper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_model = create_block_swap_wrapper(
        model=original_model,
        device=device,
        max_vram_gb=8.0,
        block_size_mb=100,
        swap_mode="adaptive",
        verbose=False
    )
    
    # Test 1: Check if model_config is accessible
    print(f"\nTest 1: Accessing model_config attribute")
    print(f"Wrapped model has model_config: {hasattr(wrapped_model, 'model_config')}")
    
    if hasattr(wrapped_model, 'model_config'):
        print(f"Wrapped model.model_config.unet_config.in_channels: {wrapped_model.model_config.unet_config.in_channels}")
        print("✓ model_config attribute accessible")
    else:
        print("✗ model_config attribute NOT accessible")
        return False
    
    # Test 2: Check if other ComfyUI attributes are accessible
    print(f"\nTest 2: Accessing other ComfyUI attributes")
    comfyui_attrs = ['model_type', 'diffusion_model', 'latent_format', 
                     'conditioning_key', 'parameterization', 'scale_factor']
    
    all_accessible = True
    for attr in comfyui_attrs:
        accessible = hasattr(wrapped_model, attr)
        print(f"  {attr}: {'✓' if accessible else '✗'}")
        if not accessible:
            all_accessible = False
    
    if all_accessible:
        print("✓ All ComfyUI attributes accessible")
    else:
        print("✗ Some ComfyUI attributes NOT accessible")
        return False
    
    # Test 3: Test __getattr__ fallback
    print(f"\nTest 3: Testing __getattr__ fallback")
    try:
        # This should work via __getattr__
        unet_config = wrapped_model.model_config.unet_config
        print(f"✓ Successfully accessed wrapped_model.model_config.unet_config")
        print(f"  in_channels: {unet_config.in_channels}")
    except AttributeError as e:
        print(f"✗ Failed to access via __getattr__: {e}")
        return False
    
    # Test 4: Test forward pass
    print(f"\nTest 4: Testing forward pass")
    try:
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output = wrapped_model(test_input)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test 5: Test eval() mode
    print(f"\nTest 5: Testing eval() mode")
    try:
        wrapped_model.eval()
        print(f"✓ eval() method works")
    except Exception as e:
        print(f"✗ eval() method failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("All tests passed! BlockSwapWrapper is compatible with ComfyUI LoRA loading.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_lora_compatibility()
    sys.exit(0 if success else 1)
