import sys
from pathlib import Path
import argparse

import torch


def load_f32(path: Path, rows: int, cols: int) -> torch.Tensor:
    count = rows * cols
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path} 的字节数不匹配")
    return torch.from_file(str(path), size=count, dtype=torch.float32).reshape(rows, cols)

# 放在 load_f32() 定义之后、测试循环之前。
parser = argparse.ArgumentParser(description="对照 PyTorch 验证 CUDA Row Softmax")
parser.add_argument(
    "--bin_dir",
    type=Path,
    required=True,
    help="包含 input_*.bin 和 output_*.bin 的目录",
)
bin_dir = parser.parse_args().bin_dir.expanduser()

if not bin_dir.is_dir():
    parser.error(f"目录不存在：{bin_dir}")

shapes = [(1, 1), (3, 17), (4, 256), (5, 257), (32, 2048)]

for rows, cols in shapes:
    for mode in ("ones", "random", "negative", "shifted"):
        name = f"{mode}_{rows}x{cols}"
        x = load_f32(bin_dir / f"input_{name}.bin", rows, cols)
        actual = load_f32(bin_dir / f"output_{name}.bin", rows, cols)
        expected = torch.softmax(x, dim=-1)

        if not torch.isfinite(actual).all().item():
            raise AssertionError(f"{name} 包含 NaN 或 Inf")

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            actual.sum(dim=-1), torch.ones(rows), rtol=0, atol=1e-6
        )
        print(f"PASS {name}  max_abs_error={(actual - expected).abs().max():.3e}")

print("PASS 20/20")