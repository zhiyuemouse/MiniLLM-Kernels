import argparse
from pathlib import Path

import torch


def load_f32(path: Path, rows: int, cols: int) -> torch.Tensor:
    count = rows * cols
    expected_bytes = count * 4
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} 文件大小不匹配，预期 {expected_bytes} 字节，"
            f"实际 {actual_bytes} 字节"
        )

    return torch.from_file(
        str(path), size=count, dtype=torch.float32
    ).reshape(rows, cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 CUDA 行 Softmax 的输出")
    parser.add_argument(
        "--bin_dir", type=Path, required=True,
        help="包含 input.bin 和 output_original.bin 的目录",
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=2048)
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows 和 --cols 必须大于 0")

    x = load_f32(args.bin_dir / "input.bin", args.rows, args.cols)
    expected = torch.softmax(x, dim=-1)

    for filename in ("output_original.bin", "output_cachedexp.bin"):
        actual = load_f32(args.bin_dir / filename, args.rows, args.cols)

        if not torch.isfinite(actual).all().item():
            raise ValueError(f"{filename} 包含 NaN 或 Inf")

        row_sums = actual.sum(dim=-1)
        max_abs_error = (actual - expected).abs().max().item()
        max_row_sum_error = (row_sums - 1.0).abs().max().item()

        print(f"{filename} 最大逐元素绝对误差: {max_abs_error:.3e}")
        print(f"{filename} 最大行和误差:       {max_row_sum_error:.3e}")

        torch.testing.assert_close(
            actual, expected, rtol=1e-5, atol=1e-7,
            msg=lambda msg: f"{filename} 逐元素对照失败\n{msg}",
        )
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-6,
            msg=lambda msg: f"{filename} 行和检查失败\n{msg}",
        )

        print(f"PASS {filename} {args.rows}x{args.cols} FP32 row softmax")


if __name__ == "__main__":
    main()