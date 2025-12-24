"""
Block Swap Wrapper for VRAM Optimization

This wrapper implements intelligent block swapping to minimize VRAM usage
while maintaining performance. It dynamically monitors VRAM usage and
adjusts block sizes accordingly.

Features:
1. Dynamic VRAM monitoring and adjustment
2. Configurable block sizes and swap strategies
3. Support for different swap modes (layerwise, chunkwise, adaptive)
4. Memory usage statistics and logging
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
import gc
import time


class BlockSwapManager:
    """
    Manages block swapping operations and VRAM monitoring.
    """
    
    def __init__(
        self,
        device: torch.device,
        max_vram_gb: float = 8.0,
        block_size_mb: int = 100,
        swap_mode: str = "adaptive",
        empty_cache_freq: int = 4,
        verbose: bool = True
    ):
        """
        Args:
            device: Target GPU device
            max_vram_gb: Maximum VRAM to use (GB)
            block_size_mb: Initial block size in MB
            swap_mode: "layerwise", "chunkwise", or "adaptive"
            empty_cache_freq: How often to empty cache (every N blocks)
            verbose: Enable verbose logging
        """
        self.device = torch.device(device)
        self.max_vram_bytes = int(max_vram_gb * 1024**3)
        self.block_size_bytes = block_size_mb * 1024**2
        self.swap_mode = swap_mode
        self.empty_cache_freq = empty_cache_freq
        self.verbose = verbose
        
        self.block_counter = 0
        self.total_swaps = 0
        self.total_vram_saved = 0
        
        # Statistics
        self.stats = {
            "blocks_processed": 0,
            "swaps_performed": 0,
            "peak_vram_used": 0,
            "avg_vram_used": 0,
            "total_time": 0.0
        }
        
    def get_vram_info(self) -> Dict[str, float]:
        """Get current VRAM usage information."""
        if self.device.type != "cuda":
            return {"used_gb": 0, "free_gb": 0, "total_gb": 0}
        
        try:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            free = total - reserved
            
            # Update peak usage
            self.stats["peak_vram_used"] = max(self.stats["peak_vram_used"], allocated)
            
            return {
                "used_gb": allocated,
                "reserved_gb": reserved,
                "free_gb": free,
                "total_gb": total
            }
        except Exception:
            return {"used_gb": 0, "free_gb": 0, "total_gb": 0}
    
    def should_swap(self, block_size_bytes: Optional[int] = None) -> bool:
        """
        Determine if we should swap based on current VRAM usage.
        
        Returns:
            True if swapping is needed, False otherwise
        """
        if self.device.type != "cuda":
            return False
        
        vram_info = self.get_vram_info()
        free_vram = vram_info["free_gb"] * 1024**3
        
        # If block size is provided, check if we have enough space
        if block_size_bytes:
            return free_vram < block_size_bytes * 1.2  # 20% buffer
        
        # Otherwise use adaptive threshold
        if self.swap_mode == "adaptive":
            # Adaptive mode: swap when free VRAM is below 30%
            return free_vram < self.max_vram_bytes * 0.3
        else:
            # Conservative mode: swap when free VRAM is below 50%
            return free_vram < self.max_vram_bytes * 0.5
    
    def optimize_block_size(self, model_size_bytes: int) -> int:
        """
        Optimize block size based on model size and available VRAM.
        
        Args:
            model_size_bytes: Total size of the model in bytes
            
        Returns:
            Optimized block size in bytes
        """
        if self.device.type != "cuda":
            return self.block_size_bytes
        
        vram_info = self.get_vram_info()
        free_vram = vram_info["free_gb"] * 1024**3
        
        if self.swap_mode == "adaptive":
            # Adaptive: use 25% of free VRAM or model size / 10, whichever is smaller
            target_size = min(free_vram * 0.25, model_size_bytes / 10)
        elif self.swap_mode == "chunkwise":
            # Chunkwise: use fixed number of chunks (8)
            target_size = model_size_bytes / 8
        else:  # layerwise
            # Layerwise: use smaller blocks for finer control
            target_size = model_size_bytes / 16
        
        # Ensure block size is within reasonable bounds
        min_block = 50 * 1024**2  # 50 MB minimum
        max_block = free_vram * 0.8  # 80% of free VRAM maximum
        
        optimized = max(min_block, min(max_block, target_size))
        
        if self.verbose:
            print(f"[BlockSwap] Optimized block size: {optimized/1024**2:.1f} MB "
                  f"(free VRAM: {free_vram/1024**3:.1f} GB)")
        
        return int(optimized)
    
    def before_block(self):
        """Called before processing a block."""
        self.block_counter += 1
        self.stats["blocks_processed"] += 1
        
        if self.empty_cache_freq > 0 and self.block_counter % self.empty_cache_freq == 0:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
    
    def after_block(self):
        """Called after processing a block."""
        pass
    
    def log_stats(self):
        """Log current statistics."""
        if not self.verbose:
            return
        
        vram_info = self.get_vram_info()
        print(f"\n[BlockSwap Statistics]")
        print(f"  Blocks processed: {self.stats['blocks_processed']}")
        print(f"  Swaps performed: {self.stats['swaps_performed']}")
        print(f"  Peak VRAM used: {self.stats['peak_vram_used']:.2f} GB")
        print(f"  Current VRAM: {vram_info['used_gb']:.2f} GB used, "
              f"{vram_info['free_gb']:.2f} GB free")
        print(f"  Total VRAM saved: {self.total_vram_saved/1024**3:.2f} GB")


class BlockSwapModule(nn.Module):
    """
    Wraps a module to enable block swapping.
    """
    
    def __init__(self, module: nn.Module, manager: BlockSwapManager):
        super().__init__()
        self.module = module
        self.manager = manager
        self._is_on_gpu = False
        
        # Estimate module size
        self.module_size_bytes = self._estimate_module_size()
    
    def _estimate_module_size(self) -> int:
        """Estimate the size of the module in bytes."""
        total = 0
        for param in self.module.parameters():
            total += param.numel() * param.element_size()
        for buffer in self.module.buffers():
            total += buffer.numel() * buffer.element_size()
        return total
    
    def _move_to_gpu(self):
        """Move module to GPU if not already there."""
        if not self._is_on_gpu and self.manager.device.type == "cuda":
            self.module.to(self.manager.device)
            self._is_on_gpu = True
    
    def _move_to_cpu(self):
        """Move module back to CPU."""
        if self._is_on_gpu:
            self.module.to("cpu")
            self._is_on_gpu = False
            if self.manager.device.type == "cuda":
                torch.cuda.empty_cache()
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with automatic block swapping.
        """
        self.manager.before_block()
        
        # Check if we need to swap
        if self.manager.should_swap(self.module_size_bytes):
            self.manager.stats["swaps_performed"] += 1
            self.manager.total_vram_saved += self.module_size_bytes
            
            if self.manager.verbose:
                vram_info = self.manager.get_vram_info()
                print(f"[BlockSwap] Swapping block (size: {self.module_size_bytes/1024**2:.1f} MB, "
                      f"free VRAM: {vram_info['free_gb']:.2f} GB)")
        
        # Move to GPU for computation
        self._move_to_gpu()
        
        try:
            # Convert inputs to match module dtype
            param = next(self.module.parameters(), None)
            target_dtype = param.dtype if param is not None else None
            
            def convert(obj):
                if isinstance(obj, torch.Tensor):
                    if (target_dtype is not None and 
                        obj.is_floating_point() and 
                        obj.dtype != target_dtype):
                        return obj.to(dtype=target_dtype)
                    return obj
                if isinstance(obj, list):
                    return [convert(x) for x in obj]
                if isinstance(obj, tuple):
                    if type(obj) is tuple:
                        return tuple(convert(x) for x in obj)
                    if hasattr(obj, "_fields"):
                        return type(obj)(*(convert(x) for x in obj))
                    return obj
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            
            args = convert(args)
            kwargs = convert(kwargs)
            
            # Execute forward pass
            output = self.module(*args, **kwargs)
            
        finally:
            # Move back to CPU to free VRAM
            self._move_to_cpu()
            self.manager.after_block()
        
        return output
    
    def to(self, device):
        """Update target device."""
        self.manager.device = torch.device(device)
        return self
    
    def cpu(self):
        """Move to CPU mode."""
        self.manager.device = torch.device("cpu")
        self._move_to_cpu()
        return self
    
    def cuda(self, device: Optional[int] = None):
        """Move to CUDA mode."""
        idx = 0 if device is None else int(device)
        self.manager.device = torch.device(f"cuda:{idx}")
        return self


class BlockSwapWrapper(nn.Module):
    """
    Main wrapper that applies block swapping to a model.
    
    This wrapper is designed to be compatible with ComfyUI's model interface,
    supporting attributes like model_config that are required for LoRA loading.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_vram_gb: float = 8.0,
        block_size_mb: int = 100,
        swap_mode: str = "adaptive",
        empty_cache_freq: int = 4,
        verbose: bool = True
    ):
        """
        Args:
            model: The model to wrap
            device: Target device
            max_vram_gb: Maximum VRAM to use (GB)
            block_size_mb: Initial block size in MB
            swap_mode: "layerwise", "chunkwise", or "adaptive"
            empty_cache_freq: How often to empty cache
            verbose: Enable verbose logging
        """
        super().__init__()
        # Store model directly in __dict__ to avoid __getattr__ recursion
        self.__dict__['model'] = model
        self.manager = BlockSwapManager(
            device=device,
            max_vram_gb=max_vram_gb,
            block_size_mb=block_size_mb,
            swap_mode=swap_mode,
            empty_cache_freq=empty_cache_freq,
            verbose=verbose
        )
        
        # Keep model on CPU initially
        model.to("cpu")
        model.eval()
        
        # Apply block swapping to appropriate modules
        self._apply_wrapping()
        
        # Forward model_config and other ComfyUI attributes
        self._forward_comfyui_attributes()
        
        # Log initial stats
        if verbose:
            self._log_initial_stats()
    
    def _forward_comfyui_attributes(self):
        """Forward ComfyUI-specific attributes from wrapped model."""
        # Forward model_config if it exists
        if hasattr(self.model, 'model_config'):
            self.model_config = self.model.model_config
        
        # Forward other common ComfyUI attributes
        comfyui_attrs = [
            'model_config', 'model_type', 'diffusion_model', 'latent_format',
            'vae', 'conditioning_key', 'parameterization', 'scale_factor',
            'disable_unet_model_creation', 'unet_config', 'adm_in_channels'
        ]
        
        for attr in comfyui_attrs:
            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))
    
    def _estimate_model_size(self) -> int:
        """Estimate total model size in bytes."""
        total = 0
        for param in self.model.parameters():
            total += param.numel() * param.element_size()
        for buffer in self.model.buffers():
            total += buffer.numel() * buffer.element_size()
        return total
    
    def _log_initial_stats(self):
        """Log initial model statistics."""
        model_size_bytes = self._estimate_model_size()
        vram_info = self.manager.get_vram_info()
        
        print(f"\n[BlockSwap Wrapper Initialized]")
        print(f"  Model size: {model_size_bytes/1024**3:.2f} GB")
        print(f"  Target device: {self.manager.device}")
        print(f"  Max VRAM: {self.manager.max_vram_bytes/1024**3:.1f} GB")
        print(f"  Swap mode: {self.manager.swap_mode}")
        print(f"  Current VRAM: {vram_info['used_gb']:.2f} GB used, "
              f"{vram_info['free_gb']:.2f} GB free")
    
    def _apply_wrapping(self):
        """Apply block swapping to model modules."""
        # Estimate model size for block optimization
        model_size_bytes = self._estimate_model_size()
        optimized_block_size = self.manager.optimize_block_size(model_size_bytes)
        self.manager.block_size_bytes = optimized_block_size
        
        # Wrap key modules for block swapping
        self._wrap_module_list("blocks")
        self._wrap_single_modules()
    
    def _wrap_module_list(self, attr_name: str):
        """Wrap a ModuleList or list of modules."""
        if not hasattr(self.model, attr_name):
            return
        
        modules = getattr(self.model, attr_name)
        
        if isinstance(modules, nn.ModuleList):
            for i in range(len(modules)):
                modules[i] = BlockSwapModule(modules[i], self.manager)
        elif isinstance(modules, (list, tuple)):
            new_modules = []
            for module in modules:
                if isinstance(module, nn.Module):
                    new_modules.append(BlockSwapModule(module, self.manager))
                else:
                    new_modules.append(module)
            
            try:
                setattr(self.model, attr_name, nn.ModuleList(new_modules))
            except Exception:
                setattr(self.model, attr_name, new_modules)
    
    def _wrap_single_modules(self):
        """Wrap individual important modules."""
        important_modules = [
            "patch_embedding", "time_embedding", "time_projection",
            "text_embedding", "rope_position_embedding", "img_emb",
            "head", "norm", "final_layer"
        ]
        
        for attr_name in important_modules:
            if hasattr(self.model, attr_name):
                module = getattr(self.model, attr_name)
                if isinstance(module, nn.Module):
                    setattr(self.model, attr_name, BlockSwapModule(module, self.manager))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        start_time = time.time()
        
        try:
            output = self.model(*args, **kwargs)
        finally:
            self.manager.stats["total_time"] += time.time() - start_time
        
        return output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Forward attribute access to wrapped model if not found in wrapper.
        This enables compatibility with ComfyUI's model interface.
        """
        # First check if attribute exists in wrapper
        if name in self.__dict__:
            return self.__dict__[name]
        
        # Special handling for 'model' attribute
        if name == 'model':
            # Try to get it from __dict__ first
            if 'model' in self.__dict__:
                return self.__dict__['model']
            # Otherwise use object.__getattribute__
            try:
                return object.__getattribute__(self, 'model')
            except AttributeError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Then check if attribute exists in wrapped model
        # Use object.__getattribute__ to avoid recursion
        try:
            model = object.__getattribute__(self, 'model')
            if hasattr(model, name):
                return getattr(model, name)
        except AttributeError:
            pass
        
        # If attribute not found, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def to(self, device):
        """Update target device."""
        self.manager.device = torch.device(device)
        return self
    
    def cpu(self):
        """Move to CPU mode."""
        self.manager.device = torch.device("cpu")
        self.model.to("cpu")
        return self
    
    def cuda(self, device: Optional[int] = None):
        """Move to CUDA mode."""
        idx = 0 if device is None else int(device)
        self.manager.device = torch.device(f"cuda:{idx}")
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.manager.stats.copy()
        stats.update(self.manager.get_vram_info())
        return stats
    
    def print_stats(self):
        """Print current statistics."""
        self.manager.log_stats()


# Factory function for easy integration
def create_block_swap_wrapper(
    model: nn.Module,
    device: torch.device,
    max_vram_gb: float = None,
    block_size_mb: int = None,
    swap_mode: str = "adaptive",
    empty_cache_freq: int = 4,
    verbose: bool = True
) -> BlockSwapWrapper:
    """
    Create a block swap wrapper for the given model.
    
    Args:
        model: The model to wrap
        device: Target device
        max_vram_gb: Maximum VRAM to use (GB). If None, auto-detect based on GPU memory.
        block_size_mb: Initial block size in MB. If None, auto-calculate based on model size.
        swap_mode: "layerwise", "chunkwise", or "adaptive"
        empty_cache_freq: How often to empty cache
        verbose: Enable verbose logging
    
    Returns:
        BlockSwapWrapper instance
    """
    # Auto-detect optimal settings if not provided
    if max_vram_gb is None and device.type == "cuda":
        try:
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            # Use 80% of total VRAM by default, leave 20% for system
            max_vram_gb = total_vram_gb * 0.8
            if verbose:
                print(f"[BlockSwap] Auto-detected {total_vram_gb:.1f}GB VRAM, using {max_vram_gb:.1f}GB max")
        except:
            max_vram_gb = 8.0  # Fallback default
    
    if max_vram_gb is None:
        max_vram_gb = 8.0  # CPU or fallback
    
    if block_size_mb is None:
        # Estimate model size
        model_size_bytes = 0
        for param in model.parameters():
            model_size_bytes += param.numel() * param.element_size()
        for buffer in model.buffers():
            model_size_bytes += buffer.numel() * buffer.element_size()
        
        model_size_gb = model_size_bytes / (1024**3)
        
        # Calculate optimal block size based on model size
        if model_size_gb < 4:
            block_size_mb = 200  # Small model
        elif model_size_gb < 8:
            block_size_mb = 300  # Medium model
        elif model_size_gb < 12:
            block_size_mb = 400  # Large model
        else:
            block_size_mb = 500  # Very large model
        
        if verbose:
            print(f"[BlockSwap] Model size: {model_size_gb:.2f}GB, using {block_size_mb}MB block size")
    
    return BlockSwapWrapper(
        model=model,
        device=device,
        max_vram_gb=max_vram_gb,
        block_size_mb=block_size_mb,
        swap_mode=swap_mode,
        empty_cache_freq=empty_cache_freq,
        verbose=verbose
    )


# Helper function for 20GB VRAM optimization
def create_optimized_20gb_wrapper(
    model: nn.Module,
    device: torch.device,
    mode: str = "balanced",
    verbose: bool = True
) -> BlockSwapWrapper:
    """
    Create an optimized block swap wrapper for 20GB VRAM systems.
    
    Args:
        model: The model to wrap
        device: Target device
        mode: "balanced" (default), "memory_saving", or "performance"
        verbose: Enable verbose logging
    
    Returns:
        BlockSwapWrapper instance
    """
    if mode == "memory_saving":
        # For maximum memory saving (if OOM issues)
        return create_block_swap_wrapper(
            model=model,
            device=device,
            max_vram_gb=14.0,
            block_size_mb=200,
            swap_mode="layerwise",
            empty_cache_freq=2,
            verbose=verbose
        )
    elif mode == "performance":
        # For maximum performance (if enough memory)
        return create_block_swap_wrapper(
            model=model,
            device=device,
            max_vram_gb=18.0,
            block_size_mb=500,
            swap_mode="chunkwise",
            empty_cache_freq=8,
            verbose=verbose
        )
    else:  # balanced (default)
        # Balanced settings for 20GB VRAM
        return create_block_swap_wrapper(
            model=model,
            device=device,
            max_vram_gb=16.0,
            block_size_mb=300,
            swap_mode="adaptive",
            empty_cache_freq=4,
            verbose=verbose
        )
