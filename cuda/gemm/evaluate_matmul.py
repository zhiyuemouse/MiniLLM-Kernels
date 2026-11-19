import argparse
from pathlib import Path

import torch


# SHAPES = ((1, 1, 1), (16, 16, 16), (17, 19, 23), (17, 35, 23))
SHAPES = ((1, 1024, 1024), (1, 35, 23))
MODES = ("ones", "random")


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


def evaluate_case(bin_dir: Path, mode: str, m: int, n: int, k: int) -> None:
    case = f"{mode}_{m}x{n}x{k}"
    a = load_f32(bin_dir / f"a_{case}.bin", m, k)
    b = load_f32(bin_dir / f"b_{case}.bin", k, n)
    actual = load_f32(bin_dir / f"c_{case}.bin", m, n)

    if mode == "ones":
        torch.testing.assert_close(a, torch.ones_like(a), rtol=0, atol=0)
        torch.testing.assert_close(b, torch.ones_like(b), rtol=0, atol=0)

    expected = a @ b
    tolerance = 0.0 if mode == "ones" else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    max_abs_error = (actual - expected).abs().max().item()
    print(f"PASS {case}  max_abs_error={max_abs_error:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="用 PyTorch 验证 CUDA GEMM")
    parser.add_argument("--bin_dir", type=Path, required=True)
    args = parser.parse_args()

    failures = 0
    for m, n, k in SHAPES:
        for mode in MODES:
            try:
                evaluate_case(args.bin_dir, mode, m, n, k)
            except (OSError, ValueError, AssertionError) as exc:
                failures += 1
                print(f"FAIL {mode}_{m}x{n}x{k}\n{exc}")

    total = len(SHAPES) * len(MODES)
    print(f"通过 {total - failures}/{total}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()