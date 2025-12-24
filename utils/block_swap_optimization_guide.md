# Block Swap Wrapper 优化指南

## 硬件配置
- **用户 VRAM**: 20GB
- **推荐设置**: 根据 20GB VRAM 优化

## 参数说明

### 1. max_vram_gb (最大 VRAM 使用量)
- **作用**: 设置块交换包装器可以使用的最大 VRAM
- **推荐值**: 16-18 GB (为系统和其他进程留出 2-4GB)
- **计算公式**: `max_vram_gb = 总 VRAM - 系统预留`
  - 20GB 总 VRAM: 推荐 16-18 GB
  - 系统预留: 2-4 GB (用于 ComfyUI、操作系统、其他模型)

### 2. block_size_mb (块大小)
- **作用**: 每个块的大小 (MB)
- **推荐值**: 200-500 MB
- **影响因素**:
  - 较小的块: 更频繁的交换，更低的内存使用，但性能较低
  - 较大的块: 较少交换，更高性能，但内存使用更高
- **优化建议**:
  - 对于 20GB VRAM: 从 300 MB 开始
  - 如果 OOM: 减小到 200 MB
  - 如果性能不足: 增加到 400-500 MB

### 3. swap_mode (交换模式)
- **adaptive** (推荐): 根据 VRAM 使用情况动态调整
- **layerwise**: 逐层交换 (最节省内存)
- **chunkwise**: 固定块大小交换 (性能较好)

## 针对 20GB VRAM 的推荐配置

### 配置 A: 平衡模式 (推荐)
```python
max_vram_gb = 16.0    # 使用 16GB，预留 4GB
block_size_mb = 300    # 300MB 块大小
swap_mode = "adaptive" # 自适应模式
```

### 配置 B: 内存优化模式 (如果仍然 OOM)
```python
max_vram_gb = 14.0    # 使用 14GB，预留 6GB
block_size_mb = 200    # 200MB 块大小
swap_mode = "layerwise" # 逐层交换 (最省内存)
```

### 配置 C: 性能优先模式 (如果有足够内存)
```python
max_vram_gb = 18.0    # 使用 18GB，预留 2GB
block_size_mb = 500    # 500MB 块大小
swap_mode = "chunkwise" # 块交换模式
```

## 优化策略

### 1. 监控 VRAM 使用
```python
# 在推理过程中监控 VRAM
import torch
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM: {allocated:.2f}GB used, {reserved:.2f}GB reserved")
```

### 2. 调整参数步骤
1. **初始设置**: 使用配置 A
2. **如果 OOM**:
   - 减小 `max_vram_gb` (如 14.0)
   - 减小 `block_size_mb` (如 200)
   - 切换到 `layerwise` 模式
3. **如果性能不足**:
   - 增加 `block_size_mb` (如 400-500)
   - 切换到 `chunkwise` 模式
   - 增加 `max_vram_gb` (如 18.0)

### 3. 模型特定优化
- **WAN 2.1/2.2 模型**: 大约 8-12GB 内存
- **双专家模型**: 需要同时加载两个模型时，需要更多内存
- **高分辨率**: 增加内存需求

## 高级优化技巧

### 1. 使用自适应模式的优势
- 自动根据可用 VRAM 调整块大小
- 在内存紧张时使用较小块
- 在内存充足时使用较大块以提高性能

### 2. 避免 OOM 的策略
1. **预留足够系统内存**: 不要使用全部 20GB
2. **监控峰值使用**: 注意推理过程中的峰值内存
3. **逐步调整**: 每次只调整一个参数
4. **测试不同分辨率**: 高分辨率需要更多内存

### 3. 性能优化
1. **增加块大小**: 减少交换次数
2. **使用 chunkwise 模式**: 更高效的交换
3. **调整 empty_cache_freq**: 减少缓存清理频率

## 故障排除

### 问题 1: 仍然 OOM
**解决方案**:
- 进一步减小 `max_vram_gb`
- 减小 `block_size_mb`
- 使用 `layerwise` 模式
- 降低分辨率或帧数

### 问题 2: 性能太慢
**解决方案**:
- 增加 `block_size_mb`
- 使用 `chunkwise` 模式
- 增加 `max_vram_gb`
- 调整 `empty_cache_freq` 为 8 或更高

### 问题 3: 不稳定
**解决方案**:
- 使用 `adaptive` 模式
- 确保足够的系统预留内存
- 监控温度和其他进程的内存使用

## 示例配置

### TurboWanModelLoader 中的使用
```python
# 在 ComfyUI 节点中使用
model_loader.load_model(
    model_name="wan_2.2_model.pth",
    attention_type="sla",
    sla_topk=0.1,
    offload_mode="block_swap",
    max_vram_gb=16.0,      # 20GB VRAM 的推荐值
    block_size_mb=300,     # 平衡性能与内存
    swap_mode="adaptive"   # 自适应优化
)
```

## 总结

对于 20GB VRAM:
- **起始配置**: `max_vram_gb=16.0`, `block_size_mb=300`, `swap_mode="adaptive"`
- **如果 OOM**: 减小参数，使用 `layerwise` 模式
- **如果性能不足**: 增加参数，使用 `chunkwise` 模式

关键是根据实际使用情况监控和调整参数。自适应模式通常能提供最佳平衡。
