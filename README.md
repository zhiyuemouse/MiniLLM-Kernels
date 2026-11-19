# MiniLLM-Kernels

面向 LLM 推理的 CUDA / Triton 算子实验。仓库包含算子实现、PyTorch 参考校验和微基准，用于观察访存、归约、分块和融合对正确性与性能的影响。

## 算子

| 目录 | 内容 |
| --- | --- |
| [cuda/reduce](cuda/reduce) | FP32 / FP16 行求和，比较 shared memory 归约与 warp shuffle |
| [cuda/softmax](cuda/softmax) | 数值稳定的 FP32 行 Softmax，实验比较重算与缓存指数值 |
| [cuda/gemm](cuda/gemm) | FP32 朴素矩阵乘法、shared memory tiling 和 `M=1` 的 GEMV 路径 |
| [cuda/triton](cuda/triton) | RMSNorm、fused residual add + RMSNorm，以及 BF16 舍入顺序对照 |

## 构建与运行

CUDA 示例需要 NVIDIA GPU、带 `nvcc` 的 CUDA Toolkit 和 CMake 3.18 或更高版本。Python 校验需要 PyTorch，Triton 实验还需要支持 CUDA 的 PyTorch 和 Triton 环境。

从仓库根目录构建。当前 CMake 配置启用 `matmul_naive`。示例中的 `86` 对应实验 GPU 的计算能力，使用其他 GPU 时按实际架构设置。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target matmul_naive --parallel

mkdir -p cuda/gemm/output
(cd cuda/gemm/output && ../../../build/matmul_naive)
```

各 CUDA 算子目录中的 `evaluate_*.py` 使用 `--bin_dir` 读取对应程序生成的二进制数据。运行前以当前 C++ 入口和 Python 脚本中的 shape、文件名为准。

运行 Triton 的 FP16 fused / unfused 对照：

```bash
python cuda/triton/rms_norm.py --benchmark-only
```

## RMSNorm 实验结果

- 独立 FP16 实验在 `4096 × 1024` 上比较完整的 residual add + RMSNorm 操作。Fused 和 unfused 的耗时中位数分别为 **194.785 µs** 和 **390.359 µs**，约 **2.00×**。
- 在 Nano-vLLM 的 Qwen3-0.6B BF16 集成实验中，batch size 1 和 2 的 CUDA Graph decode forward A/B 差异约 **0.1–0.3%**。该 workload 下两组基本持平，Triton 接入保留为默认关闭的实验选项。

## 教学参考

`cuda/` 下的 `AIINFRAGUIDE_*.cu` 和沿用现有文件名的 `AIIFRAGUIDE_onlinesoftmax.cu` 参考了开源教学项目 AIInfraGuide 的 CUDA 编程与算子优化示例。[AIInfraGuide 源项目](https://github.com/caomaolufei/AIInfraGuide)
