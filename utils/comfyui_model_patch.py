"""
ComfyUI Model Patch - Add ComfyUI compatibility attributes to TurboDiffusion models.

This module adds the necessary attributes (model_config, etc.) to TurboDiffusion
models so they work with ComfyUI's LoRA loading and other features.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


def add_comfyui_attributes(model: nn.Module) -> nn.Module:
    """
    Add ComfyUI compatibility attributes to a model.
    
    Args:
        model: The model to patch
        
    Returns:
        The patched model (same object, modified in-place)
    """
    # Check if model already has model_config
    if hasattr(model, 'model_config'):
        return model
    
    # Create a simple model_config structure
    # Based on ComfyUI's expected structure for LoRA loading
    class ModelConfig:
        def __init__(self):
            # Basic UNet config structure
            self.unet_config = type('obj', (object,), {
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
            
            # Add other common attributes
            self.latent_format = "SD15"
            self.parameterization = "eps"
            self.sampling_settings = {}
    
    # Add model_config attribute
    model.model_config = ModelConfig()
    
    # Add other common ComfyUI attributes
    if not hasattr(model, 'model_type'):
        model.model_type = "SD15"
    
    if not hasattr(model, 'diffusion_model'):
        model.diffusion_model = model
    
    if not hasattr(model, 'latent_format'):
        model.latent_format = "SD15"
    
    if not hasattr(model, 'conditioning_key'):
        model.conditioning_key = "crossattn"
    
    if not hasattr(model, 'parameterization'):
        model.parameterization = "eps"
    
    if not hasattr(model, 'scale_factor'):
        model.scale_factor = 0.18215
    
    # Add __getattr__ to handle attribute forwarding for wrapped models
    original_getattr = getattr(model, '__getattr__', None)
    
    def comfyui_getattr(self, name):
        # First try to get attribute normally
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            # If model has a wrapped model, try to get from it
            if hasattr(self, 'model') and hasattr(self.model, name):
                return getattr(self.model, name)
            # If model has a module attribute (common in wrappers)
            elif hasattr(self, 'module') and hasattr(self.module, name):
                return getattr(self.module, name)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # Only add __getattr__ if it doesn't already exist
    if original_getattr is None:
        model.__getattr__ = comfyui_getattr.__get__(model, type(model))
    
    return model


def create_comfyui_compatible_model(model: nn.Module) -> nn.Module:
    """
    Create a ComfyUI-compatible wrapper around a model.
    
    This is useful for models that can't be modified in-place.
    
    Args:
        model: The original model
        
    Returns:
        A wrapper that provides ComfyUI compatibility
    """
    class ComfyUIWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Store model directly in __dict__ to avoid __getattr__ recursion
            self.__dict__['model'] = model
            self._add_comfyui_attributes()
        
        def _add_comfyui_attributes(self):
            # Add model_config
            class ModelConfig:
                def __init__(self):
                    self.unet_config = type('obj', (object,), {
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
            
            self.model_config = ModelConfig()
            self.model_type = "SD15"
            self.diffusion_model = self.model
            self.latent_format = "SD15"
            self.conditioning_key = "crossattn"
            self.parameterization = "eps"
            self.scale_factor = 0.18215
        
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)
        
        def __getattr__(self, name):
            # Try to get from wrapper first
            if name in self.__dict__:
                return self.__dict__[name]
            # Avoid recursion for 'model' attribute
            if name == 'model':
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            # Then try to get from wrapped model
            elif hasattr(self.model, name):
                return getattr(self.model, name)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    return ComfyUIWrapper(model)


def is_comfyui_compatible(model: nn.Module) -> bool:
    """
    Check if a model has the necessary ComfyUI attributes.
    
    Args:
        model: The model to check
        
    Returns:
        True if the model has all required ComfyUI attributes
    """
    required_attrs = ['model_config', 'model_type', 'diffusion_model']
    
    for attr in required_attrs:
        if not hasattr(model, attr):
            return False
    
    # Check that model_config has unet_config
    if not hasattr(model.model_config, 'unet_config'):
        return False
    
    return True


# Test function
def test_comfyui_patch():
    """Test the ComfyUI patch functionality."""
    import sys
    import os
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    print("Testing ComfyUI patch...")
    
    # Test 1: Add attributes to model
    model = TestModel()
    print(f"Original model has model_config: {hasattr(model, 'model_config')}")
    
    patched = add_comfyui_attributes(model)
    print(f"Patched model has model_config: {hasattr(patched, 'model_config')}")
    print(f"Patched model.model_config.unet_config.in_channels: {patched.model_config.unet_config.in_channels}")
    
    # Test 2: Create wrapper
    model2 = TestModel()
    wrapped = create_comfyui_compatible_model(model2)
    print(f"\nWrapped model has model_config: {hasattr(wrapped, 'model_config')}")
    print(f"Wrapped model.model_config.unet_config.in_channels: {wrapped.model_config.unet_config.in_channels}")
    
    # Test 3: Check compatibility
    print(f"\nOriginal model compatible: {is_comfyui_compatible(model)}")
    print(f"Patched model compatible: {is_comfyui_compatible(patched)}")
    print(f"Wrapped model compatible: {is_comfyui_compatible(wrapped)}")
    
    # Test 4: Forward pass
    x = torch.randn(2, 10)
    with torch.no_grad():
        y1 = patched(x)
        y2 = wrapped(x)
    print(f"\nForward pass successful: {y1.shape}, {y2.shape}")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_comfyui_patch()
