import torch
import triton
import triton.language as tl
import time
import statistics
import argparse
import random
from nanovllm_rms_norm import RMSNorm
import inspect

@triton.jit
def rms_norm_kernel(X, Weight, Y, N, eps, BLOCK_SIZE: tl.constexpr): # 每个 Triton program 负责一行
    row = tl.program_id(axis=0) # 相当于 blockIdx.x
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = row * N + cols # 这里面就是一个program要处理的一行数据
    x = tl.load(X + row_offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    y = x * inv_rms * w
    tl.store(Y + row_offsets, y, mask=mask)

def rms_norm(x, weight, eps=1e-6):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    rms_norm_kernel[(M,)](x, weight, output, N, eps, BLOCK_SIZE)
    return output

@triton.jit
def fused_add_rms_norm_kernel(Res, X, Weight, Y, ResOut, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = row * N + cols
    res = tl.load(Res + row_offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(X + row_offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
    s = res + x
    sum_sq = tl.sum(s * s, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    y = s * inv_rms * w
    tl.store(Y + row_offsets, y, mask=mask)
    tl.store(ResOut + row_offsets, s, mask=mask)

def fused_add_rms_norm(res, x, weight, eps=1e-6):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    resout = torch.empty_like(res)
    fused_add_rms_norm_kernel[(M,)](res, x, weight, output, resout, N, eps, BLOCK_SIZE)
    return output, resout

@triton.jit
def add_residual_kernel(X, Residual, SumFP32, ResOut, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = row * N + cols
    res = tl.load(Residual + row_offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(X + row_offsets, mask=mask, other=0).to(tl.float32)
    s = res + x
    tl.store(ResOut + row_offsets, s, mask=mask)
    tl.store(SumFP32 + row_offsets, s, mask=mask)

def unfused_add_rms_norm(x, res):
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    SumFP32 = torch.empty_like(x, dtype=torch.float32)
    ResOut = torch.empty_like(res)
    add_residual_kernel[(M,)](x, res, SumFP32, ResOut, N, BLOCK_SIZE)
    return SumFP32, ResOut

@triton.jit
def nanovllm_rms_norm_kernel(Res, X, Weight, Y, ResOut, N, eps, BLOCK_SIZE: tl.constexpr, HAS_RESIDUAL: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = row * N + cols
    s = tl.load(X + row_offsets, mask=mask, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        res = tl.load(Res + row_offsets, mask=mask, other=0).to(tl.float32)
        s = res + s
    w = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
    
    sum_sq = tl.sum(s * s, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    normalized = (s * inv_rms).to(X.dtype.element_ty)
    y = normalized.to(tl.float32) * w
    tl.store(Y + row_offsets, y, mask=mask)
    if HAS_RESIDUAL:
        tl.store(ResOut + row_offsets, s, mask=mask)

def nanovllm_rms_norm(x, weight, residual=None, eps=1e-6):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    has_residual = residual is not None
    if has_residual:
        resout = torch.empty_like(x)
        nanovllm_rms_norm_kernel[(M,)](residual, x, weight, output, resout, N, eps, BLOCK_SIZE, has_residual)
        return output, resout

    nanovllm_rms_norm_kernel[(M,)](x, x, weight, output, output, N, eps, BLOCK_SIZE, has_residual)
    return output

def test_nanovllm_rms_norm_bf16():
    eps = 1e-6
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(42)

    for rows, cols in ((3, 17), (3, 1024)):
        x = torch.randn(
            rows, cols, device=device, dtype=dtype, generator=generator
        )
        residual = torch.randn(
            rows, cols, device=device, dtype=dtype, generator=generator
        )
        weight = torch.linspace(0.5, 1.5, cols, device=device).to(dtype)

        for residual_input in (None, residual):
            has_residual = residual_input is not None
            label = f"BF16 {rows}x{cols} residual={has_residual}"

            s = x.float()
            if has_residual:
                s = s + residual_input.float()

            inv_rms = torch.rsqrt(
                s.square().mean(dim=-1, keepdim=True) + eps
            )
            normalized = (s * inv_rms).to(dtype)
            expected_y = (normalized.float() * weight.float()).to(dtype)

            result = nanovllm_rms_norm(
                x, weight, residual=residual_input, eps=eps
            )
            if has_residual:
                assert isinstance(result, tuple) and len(result) == 2, label
                actual_y, actual_residual = result
                expected_residual = s.to(dtype)
                assert actual_residual.dtype == dtype, label
                assert torch.equal(actual_residual, expected_residual), (
                    f"{label} 残差输出不一致"
                )
            else:
                assert isinstance(result, torch.Tensor), label
                actual_y = result

            assert actual_y.shape == expected_y.shape, label
            assert actual_y.dtype == dtype, label
            max_error = (actual_y.float() - expected_y.float()).abs().max().item()
            mismatches = (actual_y != expected_y).sum().item()
            print(
                f"{label} max_abs_error={max_error:.3e} "
                f"mismatches={mismatches}/{expected_y.numel()}"
            )

            torch.testing.assert_close(
                actual_y, expected_y, rtol=0, atol=0, msg=label
            )
            print(f"PASS {label}")

def test_triton_rms_norm_matches_nanovllm_bf16():
    eps = 1e-6
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(42)

    for rows, cols in ((3, 17), (3, 1024), (128,1024)):
        x = torch.randn(
            rows, cols, device=device, dtype=dtype, generator=generator
        )
        residual = torch.randn(
            rows, cols, device=device, dtype=dtype, generator=generator
        )
        weight = torch.linspace(0.5, 1.5, cols, device=device).to(dtype)
        nanovllm_rms = RMSNorm(cols, eps).to(device=x.device, dtype=x.dtype).eval()
        with torch.no_grad():
            nanovllm_rms.weight.copy_(weight)
        for residual_input in (None, residual):
            has_residual = residual_input is not None
            label = f"BF16 {rows}x{cols} residual={has_residual}"
            
            with torch.inference_mode():
                compiled_result = nanovllm_rms(x, residual_input)

                if has_residual:
                    eager_forward = inspect.unwrap(RMSNorm.add_rms_forward)
                    eager_y, eager_residual = eager_forward(nanovllm_rms, x, residual_input)
                    expected_y, expected_residual = compiled_result
                else:
                    eager_forward = inspect.unwrap(RMSNorm.rms_forward)
                    eager_y = eager_forward(nanovllm_rms, x)
                    expected_y = compiled_result

            # result = nanovllm_rms_norm(
            #     x, weight, residual=residual_input, eps=eps
            # )
            if has_residual:
                result = fused_add_rms_norm(residual, x, weight, eps)
            else:
                result = rms_norm(x, weight, eps)
            
            if has_residual:
                assert isinstance(result, tuple) and len(result) == 2, label
                actual_y, actual_residual = result
                assert actual_residual.dtype == dtype, label
                assert torch.equal(actual_residual, expected_residual), (
                    f"{label} 残差输出不一致"
                )
            else:
                assert isinstance(result, torch.Tensor), label
                actual_y = result

            assert actual_y.shape == expected_y.shape, label
            assert actual_y.dtype == dtype, label
            max_error = (actual_y.float() - expected_y.float()).abs().max().item()
            mismatches = (actual_y != expected_y).sum().item()
            print(
                f"{label} max_abs_error={max_error:.3e} "
                f"mismatches={mismatches}/{expected_y.numel()}"
            )

            s = x.float() if residual_input is None else x.float() + residual_input.float()
            normalized = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + eps)

            rounded_first = (normalized.to(dtype).float() * weight.float()).to(dtype)
            rounded_last = (normalized * weight.float()).to(dtype)
            if rows == 128 and cols == 1024 and has_residual:
                triton_y = actual_y.detach().cpu()
                compiled_y = expected_y.detach().cpu()
                late_y = rounded_last.detach().cpu()

                coords = (triton_y != compiled_y).nonzero(as_tuple=False)

                for row, col in coords.tolist():
                    t = triton_y[row, col]
                    c = compiled_y[row, col]
                    ref = late_y[row, col]

                    print(f"[{row}, {col}]")
                    print(f"  Triton    {t.item():.9g}")
                    print(f"  compiled  {c.item():.9g}")
                    print(f"  后舍入参考  {ref.item():.9g}")

                    for name, value in (("Triton", t), ("compiled", c)):
                        if torch.equal(value, ref):
                            print(f"  {name} 与参考一致")
                            continue

                        # 在 BF16 精度下，从参考值朝实际值走一步。
                        neighbor = torch.nextafter(ref, value)
                        print(f"  {name} 与参考恰好相邻: {torch.equal(value, neighbor)}")

            for name, output in (
                ("Triton", actual_y),
                ("Nano-vLLM eager", eager_y),
                ("Nano-vLLM compiled", expected_y),
            ):
                assert output.shape == x.shape and output.dtype == dtype, (label, name)

                for rounding, ref in (
                    ("先舍入", rounded_first),
                    ("后舍入", rounded_last),
                ):
                    diff = (output.float() - ref.float()).abs()
                    mismatches = (output != ref).sum().item()
                    print(
                        f"{label} | {name} / {rounding} | "
                        f"max_abs_error={diff.max().item():.3e} | "
                        f"mismatches={mismatches}/{ref.numel()}"
                    )

            if has_residual:
                ref_residual = s.to(dtype)
                for name, output in (
                    ("Triton", actual_residual),
                    ("Nano-vLLM eager", eager_residual),
                    ("Nano-vLLM compiled", expected_residual),
                ):
                    matches = output.dtype == dtype and torch.equal(output, ref_residual)
                    print(f"{label} | {name} | residual_exact={matches}")

def test_fused_and_unfused_add_rms_norm_match_reference():
    print("=================== test_fused_and_unfused_add_rms_norm_match_reference ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        eps = 1e-6
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        res = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, device="cuda", dtype=torch.float16)
        grid = (M,)
        block_size = triton.next_power_of_2(N)
        y_fused = torch.empty_like(x)
        res_fused = torch.empty_like(res)

        # 非融合路径的中间缓冲区和输出
        s_fp32 = torch.empty_like(x, dtype=torch.float32)
        y_unfused = torch.empty_like(x)
        res_unfused = torch.empty_like(res)

        def launch_fused():
            fused_add_rms_norm_kernel[grid](
                res, x, weight, y_fused, res_fused, N, eps, block_size
            )

        def launch_unfused():
            add_residual_kernel[grid](
                res, x, s_fp32, res_unfused, N, block_size
            )
            rms_norm_kernel[grid](
                s_fp32, weight, y_unfused, N, eps, block_size
            )
        launch_fused()
        launch_unfused()

        s_ref = res.float() + x.float()
        inv_rms = torch.rsqrt(s_ref.square().mean(dim=-1, keepdim=True) + eps)
        ref_y = (s_ref * inv_rms * weight.float()).to(x.dtype)
        ref_res = s_ref.to(res.dtype)

        torch.testing.assert_close(y_fused, ref_y, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y_unfused, ref_y, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(res_fused, ref_res, atol=0, rtol=0)
        torch.testing.assert_close(res_unfused, ref_res, atol=0, rtol=0)

        print(f"test M={M} N={N} dtype={x.dtype} PASS")

def test_rms_norm_fp32():
    print("=================== test_rms_norm_fp32 ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        eps = 1e-6
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        weight = torch.randn(N, device="cuda", dtype=torch.float32)
        triton_output = rms_norm(x, weight, eps)

        ref_output = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight
        torch.testing.assert_close(triton_output, ref_output, atol=1e-5, rtol=1e-5)
        print(f"test M={M} N={N} PASS")

def test_rms_norm_fp16():
    print("=================== test_rms_norm_fp16 ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        eps = 1e-6
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, device="cuda", dtype=torch.float16)
        triton_output = rms_norm(x, weight, eps)

        x_fp32 = x.float()
        weight_fp32 = weight.float()
        inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        ref_output = (x_fp32 * inv_rms * weight_fp32).to(x.dtype)
        torch.testing.assert_close(triton_output, ref_output, atol=1e-4, rtol=1e-4)
        print(f"test M={M} N={N} PASS")

def test_rms_norm_fp16_large_input():
    print("=================== test_rms_norm_fp16_large_input ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        eps = 1e-6
        x = torch.ones(M, N, device="cuda", dtype=torch.float16) * 300
        weight = torch.ones(N, device="cuda", dtype=torch.float16)
        triton_output = rms_norm(x, weight, eps)

        x_fp32 = x.float()
        weight_fp32 = weight.float()
        inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        ref_output = (x_fp32 * inv_rms * weight_fp32).to(x.dtype)
        torch.testing.assert_close(triton_output, ref_output, atol=1e-5, rtol=1e-5)
        print(f"test M={M} N={N} PASS")

def test_fused_add_rms_norm():
    print("=================== test_fused_add_rms_norm ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        eps = 1e-6
        x = torch.ones(M, N, device="cuda", dtype=torch.float16) * 300
        res = torch.ones(M, N, device="cuda", dtype=torch.float16) * 2
        weight = torch.ones(N, device="cuda", dtype=torch.float16)
        triton_output, triton_res = fused_add_rms_norm(res, x, weight, eps)

        x_fp32 = x.float() + res.float()
        weight_fp32 = weight.float()
        inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        ref_output = (x_fp32 * inv_rms * weight_fp32).to(x.dtype)
        ref_res = x_fp32.to(x.dtype)
        torch.testing.assert_close(triton_output, ref_output, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(triton_res, ref_res, atol=1e-5, rtol=1e-5)
        print(f"test M={M} N={N} PASS")

def test_fused_add_rms_norm_random():
    print("=================== test_fused_add_rms_norm_random ===================")
    test_list = [(3, 17), (3, 1024)]
    for (M, N) in test_list:
        for torch_type in [torch.float16, torch.float32]:
            eps = 1e-6
            x = torch.randn(M, N, device="cuda", dtype=torch_type)
            res = torch.randn(M, N, device="cuda", dtype=torch_type)
            weight = torch.randn(N, device="cuda", dtype=torch_type)
            triton_output, triton_res = fused_add_rms_norm(res, x, weight, eps)

            x_fp32 = x.float() + res.float()
            weight_fp32 = weight.float()
            inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
            ref_output = (x_fp32 * inv_rms * weight_fp32).to(x.dtype)
            ref_res = x_fp32.to(x.dtype)
            torch.testing.assert_close(triton_output, ref_output, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(triton_res, ref_res, atol=1e-5, rtol=1e-5)
            print(f"test M={M} N={N} PASS")

def test_fused_add_rms_norm_fp16_rounding():
    print("=================== test_fused_add_rms_norm_fp16_rounding ===================")
    eps = 1e-6
    kwargs = {"device": "cuda", "dtype": torch.float16}
    x = torch.tensor([[1.0, 1.0]], **kwargs)
    res = torch.tensor([[2**-11, 0.0]], **kwargs)
    weight = torch.ones(2, **kwargs)

    def normalize(s):
        inv_rms = torch.rsqrt(s.square().mean(dim=-1, keepdim=True) + eps)
        return (s * inv_rms * weight.float()).to(x.dtype)

    s_fp32 = x.float() + res.float()
    expected_res = s_fp32.to(x.dtype)

    expected_y = normalize(s_fp32)
    early_rounded_y = normalize(expected_res.float())

    assert not torch.equal(expected_y, early_rounded_y), (
        "测试输入必须能区分两种舍入顺序"
    )

    actual_y, actual_res = fused_add_rms_norm(res, x, weight, eps)
    torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
    torch.testing.assert_close(actual_res, expected_res, rtol=0, atol=0)
    print("test_fused_add_rms_norm_fp16_rounding PASS")

def benchmark_rms_norm_fp16():
    print("=================== benchmark_rms_norm_fp16 ===================")
    M, N = 4096, 1024
    eps = 1e-6
    x = torch.ones(M, N, device="cuda", dtype=torch.float16) * 300
    weight = torch.ones(N, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(N)

    def launch():
        rms_norm_kernel[(M,)](x, weight, y, N, eps, block_size)

    # 1. 首次调用，用主机时钟
    torch.cuda.synchronize() # 等输入初始化完成
    t0 = time.perf_counter() # 开始记录主机时间
    launch() # 启动 RMSNorm kernel 一次
    torch.cuda.synchronize() # 同步 CUDA
    first_ms = (time.perf_counter() - t0) * 1000 # 单位 ms

    # 用同一份输入和权重，以 FP32 计算 PyTorch 参考结果
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    inv_rms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    ref_output = (x_fp32 * inv_rms * weight_fp32).to(x.dtype)
    torch.testing.assert_close(y, ref_output, atol=1e-5, rtol=1e-5) # 将参考结果转回 FP16，并与 kernel 输出比较

    # 预热启动 20 次
    for _ in range(20):
        launch()
    torch.cuda.synchronize()
    # 重复 5 批：
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples_us = []
    for _ in range(5):
        start.record()
        for _ in range(100):
            launch()
        stop.record()
        stop.synchronize()
        samples_us.append(start.elapsed_time(stop) * 1000 / 100)
    print("首次调用(ms)", first_ms)
    print("5 批样本(µs/launch)", samples_us)
    print("稳态中位数(µs/launch)", statistics.median(samples_us))

def benchmark_fused_vs_unfused_add_rms_norm_fp16():
    print("=================== benchmark_fused_vs_unfused_add_rms_norm_fp16 ===================")
    M, N = 4096, 1024
    eps = 1e-6
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    res = torch.randn(M, N, device="cuda", dtype=torch.float16)
    weight = torch.randn(N, device="cuda", dtype=torch.float16)
    grid = (M,)
    block_size = triton.next_power_of_2(N)
    y_fused = torch.empty_like(x)
    res_fused = torch.empty_like(res)

    # 非融合路径的中间缓冲区和输出
    s_fp32 = torch.empty_like(x, dtype=torch.float32)
    y_unfused = torch.empty_like(x)
    res_unfused = torch.empty_like(res)

    def launch_fused():
        fused_add_rms_norm_kernel[grid](res, x, weight, y_fused, res_fused, N, eps, block_size)

    def launch_unfused():
        add_residual_kernel[grid](res, x, s_fp32, res_unfused, N, block_size)
        rms_norm_kernel[grid](s_fp32, weight, y_unfused, N, eps, block_size)

    # 同一份输入，验证两条路径的两个输出。
    launch_fused()
    launch_unfused()

    s_ref = x.float() + res.float()
    inv_rms = torch.rsqrt(s_ref.square().mean(dim=-1, keepdim=True) + eps)
    ref_y = (s_ref * inv_rms * weight.float()).to(x.dtype)
    ref_res = s_ref.to(res.dtype)

    for name, y, res_out in (
        ("fused", y_fused, res_fused),
        ("unfused", y_unfused, res_unfused),
    ):
        torch.testing.assert_close(y, ref_y, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(res_out, ref_res, atol=0, rtol=0)
        print(f"PASS {name} {M}x{N} FP16")

    # 预热
    for _ in range(20):
        launch_fused()
        launch_unfused()
    torch.cuda.synchronize()

    repeats = 100
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    def time_batch_us(launch):
        start.record()
        for _ in range(repeats):
            launch()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1000 / repeats

    # 20 对样本，两种先后顺序各 10 对，固定种子打乱。
    orders = ["fused"] * 10 + ["unfused"] * 10
    random.Random(42).shuffle(orders)
    samples = []

    for first in orders:
        if first == "fused":
            fused_us = time_batch_us(launch_fused)
            unfused_us = time_batch_us(launch_unfused)
        else:
            unfused_us = time_batch_us(launch_unfused)
            fused_us = time_batch_us(launch_fused)
        samples.append((first, fused_us, unfused_us))

    print("原始配对样本 (first, fused, unfused)，单位 µs/次完整操作")
    for sample in samples:
        print(sample)

    print(f"fused 中位数: {statistics.median(f for _, f, _ in samples):.3f} µs")
    print(f"unfused 中位数: {statistics.median(u for _, _, u in samples):.3f} µs")

    paired_median = statistics.median(u - f for _, f, u in samples)
    print(f"配对差中位数 (unfused - fused): {paired_median:+.3f} µs")

    for first in ("fused", "unfused"):
        group_median = statistics.median(
            u - f for order, f, u in samples if order == first
        )
        print(f"{first} 先测，配对差中位数: {group_median:+.3f} µs")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="跳过小形状回归测试，直接运行 FP16 benchmark",
    )
    args = parser.parse_args()

    if not args.benchmark_only:
        # test_rms_norm_fp32()
        # test_rms_norm_fp16()
        # test_rms_norm_fp16_large_input()
        # test_fused_add_rms_norm()
        # test_fused_add_rms_norm_random()
        # test_fused_add_rms_norm_fp16_rounding()
        # test_fused_and_unfused_add_rms_norm_match_reference()
        # test_nanovllm_rms_norm_bf16()
        test_triton_rms_norm_matches_nanovllm_bf16()
    else:
        # benchmark_rms_norm_fp16()
        benchmark_fused_vs_unfused_add_rms_norm_fp16()