# ComfyUI TurboDiffusion I2V with Block Swap Optimization

**Forked from**: [anveshane/Comfyui_turbodiffusion](https://github.com/anveshane/Comfyui_turbodiffusion)

This is an enhanced fork of the original ComfyUI TurboDiffusion I2V custom node, specifically focused on VRAM optimization through intelligent block swapping technology.

## 🚀 Key Differences from Original

### 1. **Block Swap Optimization (New Feature)**
- **Intelligent VRAM Management**: Dynamic block swapping to minimize VRAM usage
- **Three Swap Modes**: `adaptive` (recommended), `layerwise`, `chunkwise`
- **Auto-Detection**: Automatically detects GPU memory and optimizes parameters
- **20GB VRAM Optimized**: Specialized optimization for 20GB VRAM systems

### 2. **Enhanced Memory Management**
- **Multiple Offload Modes**: `block_swap`, `comfy_native`, `layerwise_gpu`, `cpu_only`
- **Dynamic Adjustment**: Real-time VRAM monitoring and adjustment
- **Reduced OOM Errors**: Significantly lower memory usage for large models

### 3. **Improved Import System**
- **Fixed Import Issues**: Resolved `turbodiffusion_vendor` import problems
- **Better Error Handling**: Detailed error messages and debugging information
- **Vendor Module Support**: Proper handling of vendored TurboDiffusion code

### 4. **Documentation & Guides**
- **Optimization Guide**: Detailed guide for different VRAM sizes
- **Performance Tuning**: Step-by-step optimization instructions
- **Troubleshooting**: Enhanced troubleshooting section

## 📋 Original Features (Preserved)
- Complete I2V pipeline with dual-expert sampling
- SLA attention optimization (2-3x faster inference)
- Support for quantized .pth models
- Automatic model loading/offloading
- Vendored code (no external TurboDiffusion installation required)

## Features

- ✅ **Complete I2V Pipeline**: Single node handles text encoding, VAE encoding, dual-expert sampling, and decoding
- ✅ **SLA Attention**: 2-3x faster inference with Sparse Linear Attention optimization
- ✅ **Quantized Models**: Supports int8 block-wise quantized .pth models
- ✅ **Dual-Expert Sampling**: Automatic switching between high/low noise models
- ✅ **Memory Management**: Automatic model loading/offloading for efficient VRAM usage
- ✅ **Vendored Code**: No external TurboDiffusion installation required

## Requirements

- **GPU**: NVIDIA RTX 3090/4090 or better (12GB+ VRAM)
- **Software**: Python >= 3.9, PyTorch >= 2.0, ComfyUI

## Installation

1. Navigate to ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/freezelion/Comfyui_turbodiffusion_blockswap.git
```

3. Restart ComfyUI

## Required Models

Download and place in your ComfyUI models directories:

### 1. Diffusion Models (`ComfyUI/models/diffusion_models/`)
- `TurboWan2.2-I2V-A14B-high-720P-quant.pth`
- `TurboWan2.2-I2V-A14B-low-720P-quant.pth`

Download from: https://huggingface.co/thu-ml/TurboWan2.2-I2V-A14B

### 2. VAE (`ComfyUI/models/vae/`)
- `wan_2.1_vae.safetensors` (or `.pth`)

### 3. Text Encoder (`ComfyUI/models/clip/` or `text_encoders/`)
- `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (or `.pth`)

## Workflow

The workflow uses 8 nodes total:

1. **TurboWanModelLoader** → Load high noise model (.pth with SLA attention)
2. **TurboWanModelLoader** → Load low noise model (.pth with SLA attention)
3. **CLIPLoader** → Load umT5-xxl text encoder
4. **CLIPTextEncode** → Create text prompt
5. **TurboWanVAELoader** → Load Wan2.1 VAE (video VAE with temporal support)
6. **LoadImage** → Load starting image
7. **TurboDiffusionI2VSampler** → Complete inference (samples 77 frames in ~60-90s)
8. **TurboDiffusionSaveVideo** → Save as MP4/GIF/WebM

See `turbowan_workflow.json` for a complete workflow.

## Node Reference

### TurboWanModelLoader
Loads quantized .pth TurboDiffusion models with SLA attention optimization.

**Inputs**:
- `model_name`: Model file from diffusion_models/
- `attention_type`: "sla" (recommended), "sagesla" (requires SpargeAttn), or "original"
- `sla_topk`: Top-k ratio for sparse attention (0.1 default)

**Outputs**:
- `MODEL`: Loaded TurboDiffusion model

### TurboWanVAELoader
Loads Wan2.1 VAE with video encoding/decoding support.

**Inputs**:
- `vae_name`: VAE file from models/vae/ folder

**Outputs**:
- `VAE`: Wan2pt1VAEInterface object with temporal support

**Note**: This is NOT the same as ComfyUI's standard VAELoader. The Wan VAE handles video frames (B, C, T, H, W) with temporal compression, while standard VAEs only handle images (B, C, H, W).

### TurboDiffusionI2VSampler
Complete I2V inference with dual-expert sampling.

**Inputs**:
- `high_noise_model`: High noise expert from TurboWanModelLoader
- `low_noise_model`: Low noise expert from TurboWanModelLoader
- `conditioning`: Text conditioning from CLIPTextEncode
- `vae`: VAE from VAELoader
- `image`: Starting image
- `num_frames`: Frames to generate (must be 8n+1, e.g., 49, 77, 121)
- `num_steps`: Sampling steps (1-4, recommended: 4)
- `resolution`: "480", "480p", "512", "720", "720p" (see note below)
- `aspect_ratio`: 16:9, 9:16, 4:3, 3:4, 1:1
- `boundary`: Timestep for model switching (0.9 recommended)
- `sigma_max`: Initial sigma for rCM (200 recommended)
- `seed`: Random seed
- `use_ode`: ODE vs SDE sampling (false = SDE recommended)

**Outputs**:
- `frames`: Generated video frames (B*T, H, W, C)

**Resolution Note**:
- `"480"`: 480×480 (1:1), 640×480 (4:3), etc. - **Lower VRAM**
- `"480p"`: 640×640 (1:1), 832×480 (16:9), etc. - Higher VRAM
- For low VRAM (8-12GB): Use `"480"` with 49 frames
- For medium VRAM (16GB): Use `"480p"` with 77 frames or `"720p"` with 49 frames
- For high VRAM (24GB+): Use `"720p"` with 77+ frames

**How it works**:
1. Extracts text embedding from conditioning
2. Encodes start image with VAE
3. Creates conditioning dict with mask and encoded latents
4. Initializes noise with seed
5. Loads high_noise_model → samples steps 0 to boundary → offloads
6. Loads low_noise_model → samples steps boundary to num_steps → offloads
7. Decodes final latents with VAE
8. Returns frames in ComfyUI IMAGE format

### TurboDiffusionSaveVideo
Saves frame sequence as video file.

**Inputs**:
- `frames`: Video frames from sampler
- `filename_prefix`: Output filename prefix
- `fps`: Frames per second (24 default)
- `format`: "mp4", "gif", or "webm"
- `quality`: Compression quality (8 default)
- `loop`: Whether to loop (for GIF)

## Performance

With SLA attention on RTX 3090:
- 720p, 77 frames, 4 steps: ~60-90 seconds
- 2-3x faster than original attention
- ~12-15GB VRAM usage with automatic offloading

## Technical Details

### Architecture
- **Models**: TurboDiffusion Wan2.2-A14B (i2v, 14B parameters)
- **Quantization**: Block-wise int8 with automatic dequantization
- **Attention**: SLA (Sparse Linear Attention) for 2-3x speedup
- **Sampling**: rCM (Rectified Consistency Model) with dual-expert switching
- **VAE**: Wan2.1 VAE (16 channel latents)
- **Text Encoder**: umT5-xxl

### Dual-Expert Sampling
1. **High Noise Model** (steps 0 → boundary): Generates coarse motion and structure
2. **Low Noise Model** (steps boundary → num_steps): Refines details and quality
3. **Boundary** (default 0.9): Switches at 90% of sampling (e.g., step 3.6 out of 4)

### Memory Management

**ComfyUI Integration:**
- VAE wrapped with ComfyUI-compatible device management
- Automatic loading/offloading integrated with ComfyUI's model management system
- Calls `comfy.model_management.unload_all_models()` before VAE encoding
- VAE automatically moves to GPU for encoding/decoding, then returns to CPU

**Manual Management:**
- Diffusion models start on CPU
- Only one diffusion model on GPU at a time during sampling
- Automatic offloading after each sampling stage
- Text embeddings kept on CPU until needed for conditioning

**Block Swap Optimization (New Feature):**
- Intelligent block swapping to minimize VRAM usage
- Three swap modes: `adaptive` (recommended), `layerwise`, `chunkwise`
- Dynamic VRAM monitoring and adjustment
- Automatic optimization for different GPU memory sizes
- For 20GB VRAM: Uses 16GB max, 300MB block size, adaptive mode by default

**Offload Modes:**
- `comfy_native`: Uses ComfyUI's async weight offloading
- `layerwise_gpu`: Swaps blocks to GPU just-in-time
- `block_swap`: Intelligent block swapping with VRAM optimization (recommended for low VRAM)
- `cpu_only`: Runs entire forward pass on CPU (slow)

**Optimization Guide:**
See `utils/block_swap_optimization_guide.md` for detailed optimization strategies for different VRAM sizes.

## Troubleshooting

**"ModuleNotFoundError"**: Restart ComfyUI after installation

**"Model not found"**: Verify model files are in correct ComfyUI directories

**CUDA OOM**: Reduce resolution or frame count

**Slow performance**: Check that `attention_type` is "sla" (not "original")

**"TurboDiffusionI2VSampler" missing**: Ensure all vendored files were copied (turbodiffusion_vendor/)

## Credits

- [TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) by THU-ML
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous

## License

Apache 2.0 (same as TurboDiffusion)
