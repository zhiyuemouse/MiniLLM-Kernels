import argparse
from pathlib import Path

import torch

from evaluate_row_sum import load_bin


def evaluate_benchmark(bin_dir: Path) -> None:
    rows, cols = 4096, 2048
    x = load_bin(
        bin_dir / "benchmark_input.bin", rows * cols, torch.float32
    ).reshape(rows, cols)
    expected = x.sum(dim=1)

    for version in ("original", "contiguous", "shuffle"):
        actual = load_bin(
            bin_dir / f"{version}_output.bin", rows, torch.float32
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-4)
        max_error = (actual - expected).abs().max().item()
        print(f"PASS {version}  max_abs_error={max_error:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_dir", type=Path)
    evaluate_benchmark(parser.parse_args().bin_dir)