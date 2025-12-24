"""
TurboWan Model Loader - Uses TurboDiffusion's official model loading

This loader wraps TurboDiffusion's create_model() function to handle
quantized .pth models with automatic quantization support, eliminating
the need for custom dequantization code.
"""

import torch
import folder_paths
import comfy.sd
import comfy.model_management
import comfy.model_patcher
from pathlib import Path

# Import from vendored TurboDiffusion code (no external dependency needed!)
try:
    # Manually add turbodiffusion_vendor directory to sys.path
    import sys
    import os
    import importlib.util
    
    # Get the absolute path to the turbodiffusion_vendor directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    vendor_dir = os.path.join(current_dir, '..', 'turbodiffusion_vendor')
    vendor_dir = os.path.abspath(vendor_dir)
    
    print(f"[TurboDiffusion] Vendor directory: {vendor_dir}")
    
    # Check if vendor directory exists
    if not os.path.exists(vendor_dir):
        raise ImportError(f"Vendor directory does not exist: {vendor_dir}")
    
    # Add vendor directory to sys.path
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    
    # Use importlib to import turbodiffusion_vendor
    spec = importlib.util.spec_from_file_location(
        "turbodiffusion_vendor",
        os.path.join(vendor_dir, "__init__.py")
    )
    if spec is None:
        raise ImportError(f"Could not create spec for turbodiffusion_vendor")
    
    turbodiffusion_vendor = importlib.util.module_from_spec(spec)
    sys.modules["turbodiffusion_vendor"] = turbodiffusion_vendor
    spec.loader.exec_module(turbodiffusion_vendor)
    
    print(f"[TurboDiffusion] Successfully imported turbodiffusion_vendor")
    
    # Now import from the vendored modules using their absolute paths within vendor dir
    from turbodiffusion_vendor.inference.modify_model import select_model, replace_attention, replace_linear_norm
    print("[TurboDiffusion] Successfully imported modify_model functions")
    TURBODIFFUSION_AVAILABLE = True
except Exception as e:
    TURBODIFFUSION_AVAILABLE = False
    print("\n" + "="*60)
    print("ERROR: Could not import vendored TurboDiffusion code!")
    print("="*60)
    print(f"Import error: {e}")
    print(f"[TurboDiffusion] sys.path: {sys.path}")
    print(f"[TurboDiffusion] Current dir: {current_dir if 'current_dir' in locals() else 'unknown'}")
    print(f"[TurboDiffusion] Vendor dir: {vendor_dir if 'vendor_dir' in locals() else 'unknown'}")
    print("\nThis should not happen as TurboDiffusion code is vendored in the package.")
    print("Please report this issue at: https://github.com/anveshane/Comfyui_turbodiffusion/issues")
    print("="*60 + "\n")
    import traceback
    traceback.print_exc()

# Import lazy loader
from ..utils.lazy_loader import LazyModelLoader
from ..utils.timing import TimedLogger
from ..utils.comfyui_model_patch import add_comfyui_attributes, create_comfyui_compatible_model


class TurboWanModelLoader:
    """
    Load TurboDiffusion quantized models using official create_model() function.

    This loader uses TurboDiffusion's official model loading with automatic
    quantization support, providing:
    - Automatic int8 quantization handling
    - Optional SageSLA attention optimization
    - Official TurboDiffusion optimizations
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "attention_type": (["original", "sla", "sagesla"], {
                    "default": "sla",
                    "tooltip": "Attention mechanism (original=standard, sla=sparse linear attention, sagesla=requires SpargeAttn package)"
                }),
                "sla_topk": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Top-k ratio for sparse attention"
                }),
                # IMPORTANT: Keep this LAST for workflow backward-compat. ComfyUI serializes
                # widget values positionally; inserting a new widget earlier breaks old graphs.
                "offload_mode": (["comfy_native", "layerwise_gpu", "block_swap", "cpu_only"], {
                    "default": "comfy_native",
                    "tooltip": "comfy_native: ComfyUI's async weight offloading. layerwise_gpu: swaps blocks to GPU just-in-time. block_swap: intelligent block swapping with VRAM optimization. cpu_only: runs on CPU (slow)."
                }),
                "max_vram_gb": ("FLOAT", {
                    "default": 8.0,
                    "min": 2.0,
                    "max": 64.0,
                    "step": 0.5,
                    "tooltip": "Maximum VRAM to use for block_swap mode (GB)"
                }),
                "block_size_mb": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Block size for block_swap mode (MB)"
                }),
                "swap_mode": (["adaptive", "layerwise", "chunkwise"], {
                    "default": "adaptive",
                    "tooltip": "adaptive: dynamically adjusts based on VRAM. layerwise: swaps each layer. chunkwise: swaps fixed-size chunks."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    DESCRIPTION = "Load TurboDiffusion quantized models using official inference code"

    # NOTE: default must match INPUT_TYPES default for backwards-compatible workflows
    # that don't provide `offload_mode` in `widgets_values`.
    def load_model(self, model_name, attention_type="sla", sla_topk=0.1, offload_mode="comfy_native", max_vram_gb=8.0, block_size_mb=100, swap_mode="adaptive"):
        """
        Create a lazy loader for TurboDiffusion quantized model.

        This returns a lazy loader that defers actual model loading until first use.
        This eliminates upfront loading time in ComfyUI workflows.

        Args:
            model_name: Model filename from diffusion_models/
            attention_type: Type of attention (sagesla, sla, original)
            sla_topk: Top-k ratio for sparse attention
            offload_mode: Offloading strategy
            max_vram_gb: Maximum VRAM for block_swap mode
            block_size_mb: Block size for block_swap mode
            swap_mode: Swap strategy for block_swap mode

        Returns:
            Tuple containing lazy model loader
        """
        if not TURBODIFFUSION_AVAILABLE:
            raise RuntimeError(
                "Could not import vendored TurboDiffusion code!\n\n"
                "This should not happen as TurboDiffusion code is included in the package.\n"
                "Please check that all files were installed correctly and report this issue at:\n"
                "https://github.com/anveshane/Comfyui_turbodiffusion/issues\n"
            )

        model_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", model_name))

        # Use timed logger for all output
        logger = TimedLogger("ModelLoader")
        logger.section(f"Preparing Lazy Model Loader")
        logger.log(f"Model: {model_name}")
        logger.log(f"Path: {model_path}")
        logger.log(f"Attention: {attention_type}, Top-k: {sla_topk}")
        logger.log(f"Offload mode: {offload_mode}")
        if offload_mode == "block_swap":
            logger.log(f"Max VRAM: {max_vram_gb} GB, Block size: {block_size_mb} MB, Swap mode: {swap_mode}")
        logger.log(f"✓ Lazy loader created (model will load on first use)")
        print(f"{'='*60}\n")

        # Create args namespace for TurboDiffusion's create_model()
        class Args:
            def __init__(self):
                self.model = "Wan2.2-A14B"
                self.attention_type = attention_type
                self.sla_topk = sla_topk
                self.offload_mode = offload_mode
                self.max_vram_gb = max_vram_gb
                self.block_size_mb = block_size_mb
                self.swap_mode = swap_mode
                self.quant_linear = True  # Models are quantized
                self.default_norm = False

        args = Args()

        # Create lazy loader with the actual loading logic
        lazy_loader = LazyModelLoader(
            model_path=model_path,
            model_name=model_name,
            load_fn=self._load_model_impl,
            load_args=None  # Will be set below
        )

        # Set load_args with reference to lazy_loader
        lazy_loader.load_args = (args, logger, lazy_loader)

        return (lazy_loader,)

    @staticmethod
    def _load_model_impl(model_path: Path, load_args, target_device=None):
        """
        Internal method that performs the actual model loading.

        This is called by LazyModelLoader when the model is first accessed.

        Args:
            model_path: Path to model checkpoint
            load_args: Tuple of (args, logger, lazy_loader)
            target_device: Optional target device to load directly to (avoids CPU→GPU transfer)

        Returns:
            Loaded model
        """
        args, logger, lazy_loader = load_args

        # Check if lazy loader has a target device set (from .to(device) call)
        if target_device is None and hasattr(lazy_loader, '_target_device'):
            target_device = lazy_loader._target_device

        try:
            logger.log("Loading with official create_model()...")

            # Create model with meta device first (no memory allocation)
            with torch.device("meta"):
                model_arch = select_model(args.model)

            # Apply attention modifications BEFORE loading state dict
            if args.attention_type in ['sla', 'sagesla']:
                logger.log(f"Applying {args.attention_type} attention with topk={args.sla_topk}...")
                model_arch = replace_attention(model_arch, attention_type=args.attention_type, sla_topk=args.sla_topk)

            # Always load state dict to CPU first (minimal memory usage)
            # We'll handle GPU transfer after loading weights
            logger.log(f"Loading state dict to CPU...")
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)

            # Clean checkpoint wrapper keys if present
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace("_checkpoint_wrapped_module.", "")
                cleaned_state_dict[clean_key] = value
            state_dict = cleaned_state_dict
            logger.log(f"Cleaned {len(state_dict)} state dict keys")

            # Apply quantization-aware layer replacements
            logger.log(f"Applying quantization-aware replacements (quant_linear={args.quant_linear}, fast_norm={not args.default_norm})...")
            replace_linear_norm(model_arch, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)

            # Load weights
            logger.log("Loading weights into model...")
            model_arch.load_state_dict(state_dict, assign=True)

            # Keep model on CPU initially - inference code will handle GPU transfer
            # This avoids OOM during loading since model stays on CPU
            model = model_arch.cpu().eval()
            logger.log("Model loaded to CPU")

            del state_dict
            torch.cuda.empty_cache()

            # Wrap model with CPU offloading if target device is CUDA
            # This allows the model to run even if it doesn't fit entirely in VRAM
            if target_device is not None and str(target_device).startswith('cuda'):
                offload_mode = getattr(args, "offload_mode", "layerwise_gpu")
                
                if offload_mode == "cpu_only":
                    from ..utils.cpu_offload_wrapper import CPUOffloadWrapper
                    model = CPUOffloadWrapper(model, target_device)
                    logger.log("Model wrapped with CPU-only offloading (very slow)")
                elif offload_mode == "comfy_native":
                    from ..utils.comfy_native_offload import ComfyNativeOffloadCallable
                    model = ComfyNativeOffloadCallable(model, load_device=target_device)
                    logger.log("Model wrapped with ComfyUI-native async offloading")
                elif offload_mode == "block_swap":
                    try:
                        from ..utils.block_swap_wrapper import create_block_swap_wrapper, create_optimized_20gb_wrapper
                        max_vram_gb = getattr(args, "max_vram_gb", None)  # None for auto-detect
                        block_size_mb = getattr(args, "block_size_mb", None)  # None for auto-calculate
                        swap_mode = getattr(args, "swap_mode", "adaptive")
                        
                        # Check if we should use 20GB optimized wrapper
                        if max_vram_gb is None and block_size_mb is None:
                            # Try to detect if we have ~20GB VRAM
                            if target_device.type == "cuda":
                                try:
                                    total_vram_gb = torch.cuda.get_device_properties(target_device).total_memory / (1024**3)
                                    if 18 <= total_vram_gb <= 22:  # Approximately 20GB
                                        logger.log(f"Detected ~{total_vram_gb:.1f}GB VRAM, using optimized 20GB settings")
                                        model = create_optimized_20gb_wrapper(
                                            model=model,
                                            device=target_device,
                                            mode="balanced",
                                            verbose=True
                                        )
                                    else:
                                        # Use auto-detection
                                        model = create_block_swap_wrapper(
                                            model=model,
                                            device=target_device,
                                            max_vram_gb=max_vram_gb,
                                            block_size_mb=block_size_mb,
                                            swap_mode=swap_mode,
                                            empty_cache_freq=4,
                                            verbose=True
                                        )
                                except:
                                    # Fallback to auto-detection
                                    model = create_block_swap_wrapper(
                                        model=model,
                                        device=target_device,
                                        max_vram_gb=max_vram_gb,
                                        block_size_mb=block_size_mb,
                                        swap_mode=swap_mode,
                                        empty_cache_freq=4,
                                        verbose=True
                                    )
                            else:
                                # CPU or unknown device
                                model = create_block_swap_wrapper(
                                    model=model,
                                    device=target_device,
                                    max_vram_gb=max_vram_gb,
                                    block_size_mb=block_size_mb,
                                    swap_mode=swap_mode,
                                    empty_cache_freq=4,
                                    verbose=True
                                )
                        else:
                            # Use user-provided values
                            model = create_block_swap_wrapper(
                                model=model,
                                device=target_device,
                                max_vram_gb=max_vram_gb,
                                block_size_mb=block_size_mb,
                                swap_mode=swap_mode,
                                empty_cache_freq=4,
                                verbose=True
                            )
                        
                        # Log the actual settings used
                        if hasattr(model, 'manager'):
                            actual_max_vram = model.manager.max_vram_bytes / (1024**3)
                            actual_block_size = model.manager.block_size_bytes / (1024**2)
                            logger.log(f"Model wrapped with block swap (max VRAM: {actual_max_vram:.1f} GB, "
                                      f"block size: {actual_block_size:.0f} MB, mode: {swap_mode})")
                        else:
                            logger.log(f"Model wrapped with block swap (mode: {swap_mode})")
                    except ImportError as e:
                        logger.log(f"⚠️ Block swap wrapper not available, falling back to layerwise: {e}")
                        from ..utils.layerwise_gpu_offload_wrapper import LayerwiseGPUOffloadWrapper
                        model = LayerwiseGPUOffloadWrapper(model, target_device, empty_cache_every=8)
                        logger.log("Model wrapped with layerwise GPU offloading (fallback)")
                else:  # layerwise_gpu or default
                    from ..utils.layerwise_gpu_offload_wrapper import LayerwiseGPUOffloadWrapper
                    model = LayerwiseGPUOffloadWrapper(model, target_device, empty_cache_every=8)
                    logger.log("Model wrapped with layerwise GPU offloading (blocks swapped to GPU)")

            logger.log(f"✓ Successfully loaded model")
            logger.log(f"Model type: {args.model}")
            logger.log(f"Attention: {args.attention_type}")
            logger.log(f"Quantized: {args.quant_linear}")
            if target_device is not None and str(target_device).startswith('cuda'):
                logger.log(f"Offload mode: {getattr(args, 'offload_mode', 'layerwise_gpu')}")
            
            # Add ComfyUI compatibility attributes for LoRA loading
            logger.log("Adding ComfyUI compatibility attributes...")
            try:
                model = add_comfyui_attributes(model)
                logger.log("✓ Added ComfyUI attributes (model_config, etc.)")
            except Exception as e:
                logger.log(f"⚠️ Failed to add ComfyUI attributes: {e}")
                logger.log("Creating ComfyUI wrapper instead...")
                model = create_comfyui_compatible_model(model)
                logger.log("✓ Created ComfyUI wrapper")

            return model

        except Exception as e:
            logger.log(f"❌ Error loading model: {e}")
            raise RuntimeError(
                f"Failed to load TurboDiffusion model.\n"
                f"Error: {str(e)}\n\n"
                f"Make sure you have installed TurboDiffusion:\n"
                f"  pip install git+https://github.com/thu-ml/TurboDiffusion.git\n"
                f"or:\n"
                f"  uv sync\n"
            ) from e
