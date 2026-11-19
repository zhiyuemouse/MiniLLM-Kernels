import argparse
from pathlib import Path

import torch


def load_f32(path: Path, rows: int, cols: int) -> torch.Tensor:
    count = rows * cols
    expected_bytes = count * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path.name} 文件大小不匹配，"
            f"预期 {expected_bytes} 字节，实际 {actual_bytes} 字节"
        )
    return torch.from_file(
        str(path), size=count, dtype=torch.float32
    ).reshape(rows, cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="用 PyTorch 验证两版 CUDA GEMM")
    parser.add_argument("--bin_dir", type=Path, required=True)
    # parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error("m、n、k 都必须大于 0")

    a = load_f32(args.bin_dir / "benchmark_a.bin", args.m, args.k)
    b = load_f32(args.bin_dir / "benchmark_b.bin", args.k, args.n)
    expected = a.double() @ b.double()

    # 当前随机输入位于 [-1, 1]，以 K=1024 的 FP32 累加为验收目标。
    rtol, atol = 1e-5, 1e-4
    failures = 0
    for version, filename in (
        ("naive", "original_output.bin"),
        ("tiled", "tiling_output.bin"),
    ):
        try:
            actual = load_f32(
                args.bin_dir / filename, args.m, args.n
            ).double()
            if not torch.isfinite(actual).all().item():
                raise ValueError("输出包含 NaN 或 Inf")

            max_abs_error = (actual - expected).abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except (OSError, ValueError, AssertionError) as exc:
            failures += 1
            print(f"FAIL {version}\n{exc}")
        else:
            print(f"PASS {version}  max_abs_error={max_abs_error:.3e}")

    print(f"通过 {2 - failures}/2")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()