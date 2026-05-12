#!/usr/bin/env python3
"""Stress-test cumac against a battery of CUDA kernels.

For each kernel:
  - shells out to `cumac run ...`
  - reads the resulting NAME.bin
  - compares against a numpy reference
  - prints PASS/FAIL with diagnostic info on failure

Run with:
    APL_REMOTE_SOCK=/var/folders/.../T/tinygpu.sock python3 stress/run_all.py
"""
from __future__ import annotations
import os, struct, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
WORK = ROOT / ".work"
WORK.mkdir(exist_ok=True)

PASS, FAIL, SKIP = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mSKIP\033[0m"

def write_bin(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.tobytes())

def read_bin(path: Path, dtype) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=dtype).copy()

def run_cumac(cu: Path, entry: str, args: list[str], grid: str, block: str,
              cwd: Path) -> tuple[int, str, str]:
    cmd = ["cumac", "run", str(cu), entry, *args,
           "--grid", grid, "--block", block]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

def fclose(actual: np.ndarray, expected: np.ndarray, rtol=1e-4, atol=1e-4) -> tuple[bool, str]:
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    rel = diff / (np.abs(expected.astype(np.float64)) + 1e-30)
    ok = np.all((diff < atol) | (rel < rtol))
    if not ok:
        i = int(np.argmax(diff))
        return False, (f"max abs diff {diff.max():.3e} at i={i}: "
                       f"got {actual.flat[i]} expected {expected.flat[i]}")
    return True, ""

def iexact(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, str]:
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    if np.array_equal(actual, expected): return True, ""
    diffs = np.where(actual != expected)[0]
    i = int(diffs[0])
    return False, f"{len(diffs)} mismatches; first @ i={i}: got {actual.flat[i]} expected {expected.flat[i]}"

# ── individual test cases ─────────────────────────────────────────────────────

def t01_vec_mul(cwd):
    n = 4096
    a = np.arange(n, dtype=np.float32); b = np.full(n, 3.0, dtype=np.float32)
    write_bin(cwd/"a.bin", a); write_bin(cwd/"b.bin", b)
    rc,o,e = run_cumac(ROOT/"k01_vec_mul.cu", "vec_mul",
        [f"out:c:f32:{n}", f"in:a:f32:{n}:a.bin", f"in:b:f32:{n}:b.bin", f"val:n:i32:{n}"],
        "16", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"c.bin", np.float32)
    return fclose(got, a*b)

def t02_saxpy(cwd):
    n = 1024
    x = np.arange(n, dtype=np.float32); y = np.ones(n, dtype=np.float32)
    alpha = np.float32(2.5)
    alpha_bits = int(np.array(alpha).view(np.int32))
    write_bin(cwd/"x.bin", x); write_bin(cwd/"y.bin", y)
    rc,o,e = run_cumac(ROOT/"k02_saxpy.cu", "saxpy",
        [f"inout:y:f32:{n}:y.bin", f"in:x:f32:{n}:x.bin",
         f"val:n:i32:{n}", f"val:alpha_bits:i32:{alpha_bits}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"y.bin", np.float32)
    return fclose(got, alpha*x + 1.0)

def t03_reduce_atomic(cwd):
    n = 4096
    a = np.random.default_rng(0).standard_normal(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a)
    write_bin(cwd/"out.bin", np.zeros(1, dtype=np.float32))
    rc,o,e = run_cumac(ROOT/"k03_reduce_atomic.cu", "reduce_atomic",
        [f"inout:out:f32:1:out.bin", f"in:in:f32:{n}:a.bin", f"val:n:i32:{n}"],
        "16", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32)[0]
    return fclose(np.array(got), np.array(a.sum()), rtol=1e-2, atol=1e-2)

def t04_reduce_shared(cwd):
    n, B = 4096, 256
    a = np.arange(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a)
    nblocks = n // B
    rc,o,e = run_cumac(ROOT/"k04_reduce_shared.cu", "reduce_shared",
        [f"out:partials:f32:{nblocks}", f"in:in:f32:{n}:a.bin", f"val:n:i32:{n}"],
        str(nblocks), str(B), cwd)
    if rc: return False, e
    got = read_bin(cwd/"partials.bin", np.float32)
    exp = a.reshape(nblocks, B).sum(axis=1)
    return fclose(got, exp)

def t05_warp_shuffle(cwd):
    n = 1024
    a = np.arange(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a)
    rc,o,e = run_cumac(ROOT/"k05_warp_shuffle.cu", "warp_reduce",
        [f"out:out:f32:{n//32}", f"in:in:f32:{n}:a.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32)
    exp = a.reshape(-1, 32).sum(axis=1)
    return fclose(got, exp)

def t06_matmul_naive(cwd):
    N = 64
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N,N), dtype=np.float32)
    B = rng.standard_normal((N,N), dtype=np.float32)
    write_bin(cwd/"A.bin", A); write_bin(cwd/"B.bin", B)
    rc,o,e = run_cumac(ROOT/"k06_matmul_naive.cu", "matmul_naive",
        [f"out:C:f32:{N*N}", f"in:A:f32:{N*N}:A.bin", f"in:B:f32:{N*N}:B.bin", f"val:N:i32:{N}"],
        f"{N//16},{N//16}", "16,16", cwd)
    if rc: return False, e
    got = read_bin(cwd/"C.bin", np.float32).reshape(N,N)
    return fclose(got, A @ B, rtol=1e-3, atol=1e-3)

def t07_matmul_tiled(cwd):
    N = 64
    rng = np.random.default_rng(1)
    A = rng.standard_normal((N,N), dtype=np.float32)
    B = rng.standard_normal((N,N), dtype=np.float32)
    write_bin(cwd/"A.bin", A); write_bin(cwd/"B.bin", B)
    rc,o,e = run_cumac(ROOT/"k07_matmul_tiled.cu", "matmul_tiled",
        [f"out:C:f32:{N*N}", f"in:A:f32:{N*N}:A.bin", f"in:B:f32:{N*N}:B.bin", f"val:N:i32:{N}"],
        f"{N//16},{N//16}", "16,16", cwd)
    if rc: return False, e
    got = read_bin(cwd/"C.bin", np.float32).reshape(N,N)
    return fclose(got, A @ B, rtol=1e-3, atol=1e-3)

def t08_transpose(cwd):
    rows, cols = 64, 64
    rng = np.random.default_rng(2)
    A = rng.standard_normal((rows,cols), dtype=np.float32)
    write_bin(cwd/"A.bin", A)
    rc,o,e = run_cumac(ROOT/"k08_transpose.cu", "transpose",
        [f"out:out:f32:{rows*cols}", f"in:in:f32:{rows*cols}:A.bin",
         f"val:rows:i32:{rows}", f"val:cols:i32:{cols}"],
        f"{cols//32},{rows//32}", "32,32", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32).reshape(cols, rows)
    return fclose(got, A.T)

def t09_histogram(cwd):
    n = 8192
    rng = np.random.default_rng(3)
    data = rng.integers(0, 256, n, dtype=np.uint8)
    write_bin(cwd/"data.bin", data)
    write_bin(cwd/"bins.bin", np.zeros(256, dtype=np.int32))
    rc,o,e = run_cumac(ROOT/"k09_histogram.cu", "histogram",
        [f"inout:bins:i32:256:bins.bin", f"in:in:u8:{n}:data.bin", f"val:n:i32:{n}"],
        "32", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"bins.bin", np.int32)
    exp = np.bincount(data, minlength=256).astype(np.int32)
    return iexact(got, exp)

def t10_stencil5(cwd):
    R, C = 64, 64
    rng = np.random.default_rng(4)
    A = rng.standard_normal((R,C), dtype=np.float32)
    write_bin(cwd/"A.bin", A)
    write_bin(cwd/"out.bin", A)  # interior overwritten; edges kept
    rc,o,e = run_cumac(ROOT/"k10_stencil5.cu", "stencil5",
        [f"inout:out:f32:{R*C}:out.bin", f"in:in:f32:{R*C}:A.bin",
         f"val:rows:i32:{R}", f"val:cols:i32:{C}"],
        f"{C//16},{R//16}", "16,16", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32).reshape(R, C)
    exp = A.copy()
    exp[1:-1,1:-1] = 0.2 * (A[1:-1,1:-1] + A[:-2,1:-1] + A[2:,1:-1]
                            + A[1:-1,:-2] + A[1:-1,2:])
    return fclose(got, exp, rtol=1e-4, atol=1e-4)

def t11_softmax(cwd):
    N, D = 8, 1024
    rng = np.random.default_rng(5)
    X = rng.standard_normal((N,D), dtype=np.float32)
    write_bin(cwd/"X.bin", X)
    rc,o,e = run_cumac(ROOT/"k11_softmax_row.cu", "softmax_row",
        [f"out:out:f32:{N*D}", f"in:in:f32:{N*D}:X.bin", f"val:N:i32:{N}"],
        str(N), "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32).reshape(N, D)
    m = X.max(axis=1, keepdims=True)
    e2 = np.exp(X - m); exp = e2 / e2.sum(axis=1, keepdims=True)
    return fclose(got, exp.astype(np.float32), rtol=1e-3, atol=1e-4)

def t12_int_arith(cwd):
    n = 1024
    rng = np.random.default_rng(6)
    a = rng.integers(-1<<20, 1<<20, n, dtype=np.int32)
    b = rng.integers(-1<<20, 1<<20, n, dtype=np.int32)
    write_bin(cwd/"a.bin", a); write_bin(cwd/"b.bin", b)
    rc,o,e = run_cumac(ROOT/"k12_int_arith.cu", "int_arith",
        [f"out:out:i32:{n}", f"in:a:i32:{n}:a.bin", f"in:b:i32:{n}:b.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.int32)
    v = (a.astype(np.int32) * np.int32(1103515245) + b.astype(np.int32) + np.int32(12345))
    v = v ^ (a >> 3)
    exp = (v & np.int32(0x7fffffff)).astype(np.int32)
    return iexact(got, exp)

def t13_fp64(cwd):
    n = 1024
    rng = np.random.default_rng(7)
    a = rng.standard_normal(n).astype(np.float64)
    b = rng.standard_normal(n).astype(np.float64)
    write_bin(cwd/"a.bin", a); write_bin(cwd/"b.bin", b)
    rc,o,e = run_cumac(ROOT/"k13_fp64.cu", "fp64_op",
        [f"out:out:f64:{n}", f"in:a:f64:{n}:a.bin", f"in:b:f64:{n}:b.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float64)
    return fclose(got, a*b + a, rtol=1e-12, atol=1e-12)

def t14_fp16(cwd):
    n = 1024
    rng = np.random.default_rng(8)
    a = rng.standard_normal(n).astype(np.float16)
    b = rng.standard_normal(n).astype(np.float16)
    write_bin(cwd/"a.bin", a); write_bin(cwd/"b.bin", b)
    rc,o,e = run_cumac(ROOT/"k14_fp16.cu", "fp16_op",
        [f"out:out:f16:{n}", f"in:a:f16:{n}:a.bin", f"in:b:f16:{n}:b.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float16)
    exp = (a.astype(np.float32)*b.astype(np.float32) + a.astype(np.float32)).astype(np.float16)
    return fclose(got.astype(np.float32), exp.astype(np.float32), rtol=2e-2, atol=2e-2)

def t15_strided_copy(cwd):
    n = 1024; stride = 7
    a = np.arange(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a)
    rc,o,e = run_cumac(ROOT/"k15_strided_copy.cu", "strided_copy",
        [f"out:out:f32:{n}", f"in:in:f32:{n}:a.bin",
         f"val:n:i32:{n}", f"val:stride:i32:{stride}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32)
    exp = a[(np.arange(n)*stride) % n]
    return fclose(got, exp)

def t16_branchy(cwd):
    n = 1024
    rng = np.random.default_rng(9)
    a = rng.standard_normal(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a)
    rc,o,e = run_cumac(ROOT/"k16_branchy.cu", "branchy",
        [f"out:out:f32:{n}", f"in:in:f32:{n}:a.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.float32)
    t = np.arange(n) & 7
    v = a
    exp = np.empty_like(a)
    exp[t==0] = v[t==0]**2
    exp[t==1] = v[t==1] + 1
    exp[t==2] = v[t==2] - 1
    exp[t==3] = np.exp(v[t==3])
    exp[t==4] = np.log(np.abs(v[t==4]) + 1)
    exp[t==5] = (v[t==5] * np.arange(16).sum())
    exp[t==6] = np.abs(v[t==6])
    exp[t==7] = -v[t==7]
    return fclose(got, exp.astype(np.float32), rtol=2e-3, atol=2e-3)

def t17_big_grid(cwd):
    # 4M threads — pragmatic stress without 16M risk.
    n = 4 * 1024 * 1024
    rc,o,e = run_cumac(ROOT/"k17_big_grid.cu", "big_grid",
        [f"out:out:u32:{n}", f"val:n:i32:{n}"],
        str(n // 256), "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.uint32)
    exp = np.arange(n, dtype=np.uint32)
    return iexact(got, exp)

def t18_no_args(cwd):
    n = 1024
    seed_u32 = 0xdeadbeef  # tinygrad NV vals are unsigned 32-bit
    rc,o,e = run_cumac(ROOT/"k18_no_args.cu", "no_args",
        [f"out:out:i32:{n}", f"val:seed:i32:{seed_u32}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"out.bin", np.int32)
    exp = (np.arange(n, dtype=np.uint32) ^ np.uint32(seed_u32)).astype(np.int32)
    return iexact(got, exp)

def t19_inplace(cwd):
    n = 1024
    a = np.arange(n, dtype=np.float32)
    write_bin(cwd/"x.bin", a)
    rc,o,e = run_cumac(ROOT/"k19_inplace.cu", "scale2_inplace",
        [f"inout:x:f32:{n}:x.bin", f"val:n:i32:{n}"],
        "4", "256", cwd)
    if rc: return False, e
    got = read_bin(cwd/"x.bin", np.float32)
    return fclose(got, a*2)

def t20_dot(cwd):
    n, B = 4096, 256
    rng = np.random.default_rng(10)
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)
    write_bin(cwd/"a.bin", a); write_bin(cwd/"b.bin", b)
    nblocks = n // B
    rc,o,e = run_cumac(ROOT/"k20_dot.cu", "dot_blocks",
        [f"out:partials:f32:{nblocks}", f"in:a:f32:{n}:a.bin", f"in:b:f32:{n}:b.bin", f"val:n:i32:{n}"],
        str(nblocks), str(B), cwd)
    if rc: return False, e
    got = read_bin(cwd/"partials.bin", np.float32)
    exp = (a*b).reshape(nblocks, B).sum(axis=1)
    return fclose(got, exp, rtol=1e-3, atol=1e-3)

TESTS = [
    ("01 vec_mul",        t01_vec_mul),
    ("02 saxpy",          t02_saxpy),
    ("03 reduce_atomic",  t03_reduce_atomic),
    ("04 reduce_shared",  t04_reduce_shared),
    ("05 warp_shuffle",   t05_warp_shuffle),
    ("06 matmul_naive",   t06_matmul_naive),
    ("07 matmul_tiled",   t07_matmul_tiled),
    ("08 transpose",      t08_transpose),
    ("09 histogram",      t09_histogram),
    ("10 stencil5",       t10_stencil5),
    ("11 softmax_row",    t11_softmax),
    ("12 int_arith",      t12_int_arith),
    ("13 fp64",           t13_fp64),
    ("14 fp16",           t14_fp16),
    ("15 strided_copy",   t15_strided_copy),
    ("16 branchy",        t16_branchy),
    ("17 big_grid",       t17_big_grid),
    ("18 no_args",        t18_no_args),
    ("19 inplace",        t19_inplace),
    ("20 dot_blocks",     t20_dot),
]

def main() -> int:
    results = []
    for name, fn in TESTS:
        cwd = WORK / name.split()[0]
        cwd.mkdir(exist_ok=True)
        t0 = time.time()
        try:
            ok, info = fn(cwd)
        except Exception as exc:
            ok, info = False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        dt = time.time() - t0
        tag = PASS if ok else FAIL
        print(f"  {tag}  {name:24s} {dt*1e3:7.1f} ms" + (f"   {info}" if info and not ok else ""))
        results.append((name, ok, info))

    n_pass = sum(1 for _,ok,_ in results if ok)
    print(f"\n{n_pass}/{len(results)} passed")
    return 0 if n_pass == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
