import argparse
import re
from pathlib import Path

import torch


CASE_NAME = re.compile(r"(random|val)_(half|float)(?:_(\d+)x(\d+))?")
LEGACY_SHAPE = (32, 2048)  # .cu 中不带尺寸的文件保存的是最后一组数据


def load_bin(path: Path, count: int, dtype: torch.dtype) -> torch.Tensor:
    expected_bytes = count * torch.empty((), dtype=dtype).element_size()
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path.name} 大小不匹配，预期 {expected_bytes} 字节，"
            f"实际 {actual_bytes} 字节"
        )
    return torch.from_file(str(path), size=count, dtype=dtype)


def evaluate(bin_dir: Path) -> None:
    def find_cases(prefix: str) -> dict[str, Path]:
        return {
            key: path
            for path in bin_dir.glob(f"{prefix}*.bin")
            if CASE_NAME.fullmatch(key := path.stem[len(prefix):])
        }

    inputs = find_cases("input_")
    outputs = find_cases("output_")
    if not inputs and not outputs:
        raise FileNotFoundError(f"未找到符合命名规则的 .bin 文件：{bin_dir}")

    missing_outputs = sorted(inputs.keys() - outputs.keys())
    missing_inputs = sorted(outputs.keys() - inputs.keys())
    if missing_outputs or missing_inputs:
        raise ValueError(
            f"配对不完整。缺少输出：{missing_outputs}，缺少输入：{missing_inputs}"
        )

    failures = []
    for key in sorted(inputs):
        mode, precision, rows_text, cols_text = CASE_NAME.fullmatch(key).groups()
        rows, cols = (
            (int(rows_text), int(cols_text)) if rows_text else LEGACY_SHAPE
        )
        input_dtype = torch.float16 if precision == "half" else torch.float32

        try:
            x = load_bin(inputs[key], rows * cols, input_dtype).reshape(rows, cols)
            actual = load_bin(outputs[key], rows, torch.float32)
            expected = x.float().sum(dim=-1)

            if mode == "val":  # 当前 .cu 的常量输入全部为 1
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-4)

            max_error = (actual - expected).abs().max().item()
            print(f"PASS {key}  max_abs_error={max_error:.3e}")
        except (AssertionError, ValueError) as exc:
            failures.append(key)
            print(f"FAIL {key}\n{exc}")

    print(f"通过 {len(inputs) - len(failures)}/{len(inputs)} 对文件")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_dir", nargs="?", type=Path, default=Path.cwd() / "output")
    evaluate(parser.parse_args().bin_dir)