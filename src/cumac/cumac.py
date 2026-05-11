#!/usr/bin/env python3
"""
cumac — write a .cu kernel, run it on macOS via tinygrad's NV runtime.

  Compile path:  Docker (nvidia/cuda) → nvcc --cubin → cubin bytes
  Run path:      tinygrad NVProgram → TinyGPU DriverKit → eGPU

USAGE
    cumac compile FILE.cu [-a sm_86] [-o OUT.cubin]
    cumac run FILE.cu ENTRY ARG... --grid Gx[,Gy,Gz] --block Bx[,By,Bz] [-v]
    cumac analyze FILE.cu [ENTRY] [--block N] [-a sm_86]

KERNEL CONSTRAINTS
    * Declare with `extern "C"` to avoid C++ name mangling.
    * Put all pointer args BEFORE scalar (int) args.
      tinygrad's NVProgram takes bufs as positional and vals as a tuple of
      ints; the cubin ABI must agree.

ARG SYNTAX (positional, in kernel-signature order)
    in:NAME:DTYPE:N[:SOURCE]      input buffer
    out:NAME:DTYPE:N              output buffer (written back to NAME.bin)
    inout:NAME:DTYPE:N[:SOURCE]   in-place buffer
    val:NAME:i32:VALUE            integer scalar (MUST follow all buffers)

    DTYPE   = f32 | i32 | u32 | i64 | u64 | u8 | i8
    SOURCE  = random | zeros | ones | iota | <path-to-bin-file>
              (default: zeros for `out`, random for `in`/`inout`)

EXAMPLE
    cat > vec_add.cu <<'EOF'
    extern "C" __global__
    void vec_add(float *c, const float *a, const float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    EOF
    cumac run vec_add.cu vec_add \\
        out:c:f32:1024 in:a:f32:1024:iota in:b:f32:1024:iota val:n:i32:1024 \\
        --grid 4 --block 256
"""
from __future__ import annotations
import argparse, hashlib, math, os, struct, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# ── dtype table ────────────────────────────────────────────────────────────────
DTYPES = {
    "f32": np.float32, "i32": np.int32,  "u32": np.uint32,
    "i64": np.int64,   "u64": np.uint64, "i8":  np.int8, "u8": np.uint8,
}

# ── arg parsing ───────────────────────────────────────────────────────────────
@dataclass
class Arg:
    kind: str         # "in" | "out" | "inout" | "val"
    name: str
    dtype: str
    n: int            # buffer length for buffers, value for scalars
    source: str | None = None   # for buffers
    data: np.ndarray | None = None  # populated by materialize()
    buf: object = None          # HCQBuffer once allocated

    @property
    def is_buf(self) -> bool: return self.kind != "val"

def parse_arg(spec: str) -> Arg:
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(f"bad arg spec {spec!r}: expected KIND:NAME:DTYPE:...")
    kind, name, dtype = parts[0], parts[1], parts[2]
    if dtype not in DTYPES:
        raise ValueError(f"{spec}: unknown dtype {dtype!r}, want one of {list(DTYPES)}")
    if kind == "val":
        if len(parts) != 4: raise ValueError(f"{spec}: val needs KIND:NAME:DTYPE:VALUE")
        if not dtype.startswith(("i", "u")):
            raise ValueError(f"{spec}: only int scalars supported (tinygrad NVProgram vals are ints)")
        return Arg(kind, name, dtype, int(parts[3]))
    if kind not in ("in", "out", "inout"):
        raise ValueError(f"{spec}: unknown kind {kind!r}")
    if len(parts) < 4: raise ValueError(f"{spec}: buffer needs KIND:NAME:DTYPE:N[:SOURCE]")
    n = int(parts[3])
    source = parts[4] if len(parts) >= 5 else ("zeros" if kind == "out" else "random")
    return Arg(kind, name, dtype, n, source)

def materialize(arg: Arg, rng: np.random.Generator) -> None:
    dt = DTYPES[arg.dtype]
    if arg.source == "random":
        if np.issubdtype(dt, np.floating):
            arg.data = rng.standard_normal(arg.n, dtype=np.float32).astype(dt)
        else:
            info = np.iinfo(dt)
            arg.data = rng.integers(info.min, info.max, arg.n, dtype=dt, endpoint=False)
    elif arg.source == "zeros": arg.data = np.zeros(arg.n, dtype=dt)
    elif arg.source == "ones":  arg.data = np.ones(arg.n, dtype=dt)
    elif arg.source == "iota":  arg.data = np.arange(arg.n, dtype=dt)
    elif arg.source and Path(arg.source).is_file():
        raw = Path(arg.source).read_bytes()
        arg.data = np.frombuffer(raw, dtype=dt, count=arg.n).copy()
    else:
        raise ValueError(f"{arg.name}: unknown source {arg.source!r}")

# ── compile (Docker NVCC) ─────────────────────────────────────────────────────
NVCC_IMAGE = os.environ.get("CUMAC_NVCC_IMAGE", "nvidia/cuda:12.6.0-devel-ubuntu22.04")
CACHE_DIR  = Path(os.environ.get("CUMAC_CACHE", Path.home() / ".cache" / "cumac"))

def compile_cu(src: Path, arch: str = "sm_86", verbose: bool = False) -> bytes:
    src = src.resolve()
    if not src.exists(): raise FileNotFoundError(src)
    key = hashlib.sha256(src.read_bytes() + arch.encode() + NVCC_IMAGE.encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{key}.cubin"
    if cached.exists():
        if verbose: print(f"[compile] cache hit {cached}", file=sys.stderr)
        return cached.read_bytes()
    # Write cubin to a bind-mounted file rather than capturing it from stdout —
    # some nvcc images print a banner from their entrypoint that would corrupt the bytes.
    out_name = f".cumac-{key}.cubin"
    sh_cmd = f"nvcc --cubin -arch={arch} {src.name} -o /w/{out_name}"
    cmd = ["docker", "run", "--rm", "--entrypoint", "sh",
           "-v", f"{src.parent}:/w", "-w", "/w",
           NVCC_IMAGE, "-c", sh_cmd]
    if verbose: print(f"[compile] {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True)
    out_path = src.parent / out_name
    if r.returncode != 0 or not out_path.exists():
        sys.stderr.write(r.stdout.decode(errors="replace"))
        sys.stderr.write(r.stderr.decode(errors="replace"))
        raise SystemExit(f"nvcc failed (exit {r.returncode})")
    cubin = out_path.read_bytes()
    out_path.unlink()
    cached.write_bytes(cubin)
    if verbose: print(f"[compile] {len(cubin)} bytes → {cached}", file=sys.stderr)
    return cubin

# ── run (tinygrad NV runtime) ─────────────────────────────────────────────────
def parse_dim(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if not 1 <= len(parts) <= 3: raise ValueError(f"bad dim {s!r}")
    while len(parts) < 3: parts.append(1)
    return tuple(parts)  # type: ignore

def run_kernel(cubin: bytes, entry: str, kargs: list[Arg],
               grid: tuple[int,int,int], block: tuple[int,int,int],
               verbose: bool = False) -> None:
    # Validate ordering: buffers must precede scalars (tinygrad ABI).
    seen_val = False
    for a in kargs:
        if a.kind == "val": seen_val = True
        elif seen_val:
            raise SystemExit(f"arg ordering: buffer {a.name!r} appears after a scalar val. "
                             f"Reorder your kernel so all pointers come first.")

    # Import here so `cumac compile` works without tinygrad installed.
    from tinygrad import Device
    from tinygrad.runtime.ops_nv import NVProgram

    dev = Device["NV"]
    alloc = dev.allocator
    if verbose: print(f"[run] device = {dev.device}", file=sys.stderr)

    # Materialize host data, allocate GPU buffers, copy in.
    rng = np.random.default_rng(seed=0)
    bufs, vals = [], []
    for a in kargs:
        if a.kind == "val":
            vals.append(int(a.n))
            continue
        materialize(a, rng)
        assert a.data is not None
        nbytes = a.data.nbytes
        a.buf = alloc.alloc(nbytes)
        if a.kind in ("in", "inout"):
            alloc._copyin(a.buf, memoryview(a.data).cast("B"))
        bufs.append(a.buf)
        if verbose:
            print(f"[run] {a.kind:5s} {a.name:8s} {a.dtype} N={a.n} "
                  f"({nbytes} B) @ {hex(a.buf.va_addr)}", file=sys.stderr)

    # Build the program and launch.
    prog = NVProgram(dev, entry, cubin)
    if verbose:
        print(f"[run] launching {entry} grid={grid} block={block} "
              f"bufs={len(bufs)} vals={vals}", file=sys.stderr)
    t = prog(*bufs, global_size=grid, local_size=block,
             vals=tuple(vals), wait=True)
    dev.synchronize()
    if t is not None:
        print(f"[run] elapsed: {t*1e3:.3f} ms", file=sys.stderr)

    # Copy outputs back and write to disk.
    for a in kargs:
        if a.kind in ("out", "inout"):
            out = bytearray(a.data.nbytes)
            alloc._copyout(memoryview(out), a.buf)
            a.data = np.frombuffer(bytes(out), dtype=DTYPES[a.dtype]).copy()
            path = Path(f"{a.name}.bin")
            path.write_bytes(bytes(out))
            head = a.data[:min(8, a.n)]
            print(f"{a.name} ({a.dtype}×{a.n}) → {path}  head={head.tolist()}")

# ── analyze (occupancy + instruction mix from cuobjdump) ──────────────────────
# Per-SM hardware limits. Sources: CUDA C++ Programming Guide §H, "Compute
# Capabilities" tables, and the Ampere/Ada whitepapers. `warp_alloc` is the
# register-allocation granularity per warp (CC 7.x+ allocs in chunks of 256).
SM_LIMITS = {
    "sm_70": dict(threads=2048, warps=64, blocks=32, regs=65536, smem=96*1024,  warp_alloc=256),
    "sm_75": dict(threads=1024, warps=32, blocks=16, regs=65536, smem=64*1024,  warp_alloc=256),
    "sm_80": dict(threads=2048, warps=64, blocks=32, regs=65536, smem=163*1024, warp_alloc=256),
    "sm_86": dict(threads=1536, warps=48, blocks=16, regs=65536, smem=100*1024, warp_alloc=256),
    "sm_89": dict(threads=1536, warps=48, blocks=24, regs=65536, smem=100*1024, warp_alloc=256),
    "sm_90": dict(threads=2048, warps=64, blocks=32, regs=65536, smem=228*1024, warp_alloc=256),
}

# Coarse SASS opcode → category map. Categorises by opcode prefix; keys are
# checked longest-first. Doesn't try to be exhaustive — enough to spot the
# shape of a kernel (memory-heavy? FP-heavy? branchy?). Add more as needed.
SASS_CATEGORIES = [
    # (category, opcodes)
    ("mem",  ["LDG", "LDL", "LDS", "STG", "STL", "STS", "ATOM", "RED", "LDGSTS", "LDC"]),
    ("fp",   ["FFMA", "FADD", "FMUL", "FMNMX", "FSET", "FSETP", "FCHK", "FSEL",
              "DFMA", "DADD", "DMUL", "DSETP", "DMNMX",
              "HFMA2", "HADD2", "HMUL2", "HSETP2", "HMNMX2", "HFMA", "HADD", "HMUL",
              "MUFU", "RRO", "F2F", "F2I", "I2F"]),
    ("int",  ["IMAD", "IADD3", "IADD", "IMUL", "IMNMX", "ISETP", "ICMP", "ISCADD",
              "BFE", "BFI", "FLO", "POPC", "LEA", "SHF", "SHL", "SHR",
              "LOP3", "LOP", "PRMT", "VOTE", "PLOP3"]),
    ("ctrl", ["BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "KILL",
              "BSSY", "BSYNC", "BREAK", "SYNC", "WARPSYNC", "BAR", "MEMBAR",
              "DEPBAR", "ERRBAR", "NANOSLEEP"]),
    ("mov",  ["MOV", "S2R", "R2P", "P2R", "SEL", "ULDC", "UMOV", "USEL", "ULDC"]),
]

def _categorize_op(op:str) -> str:
    for cat, opcodes in SASS_CATEGORIES:
        if op in opcodes: return cat
        for prefix in opcodes:
            if op.startswith(prefix): return cat
    return "other"

def _parse_resource_usage(text:str) -> dict[str, dict[str, int]]:
    """Parse `cuobjdump --dump-resource-usage` output → {entry: {REG: n, SHARED: n, ...}}."""
    out, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Function "):
            current = line.split()[1].rstrip(":")
            out[current] = {}
        elif current and ":" in line:
            for tok in line.split():
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    try: out[current][k] = int(v)
                    except ValueError: pass
    return out

def _parse_sass(text:str, entry:str) -> list[str]:
    """Extract SASS opcodes for one function. Returns ordered list of opcode names."""
    ops, in_fn = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Function : "):
            in_fn = (s.split(":", 1)[1].strip() == entry)
            continue
        if not in_fn: continue
        # Skip the second-hex-row continuation lines (start with /* and have only a hex literal).
        if "/*" in line and "*/" in line:
            # Lines we care about look like:  /*0000*/ <maybe @P0> OPCODE  args ;
            after = line.split("*/", 1)[1].strip()
            if not after or after.startswith("/*"): continue
            toks = after.split()
            i = 0
            # Strip predicate guard like @P0 or @!P1.
            if toks and toks[i].startswith("@"): i += 1
            if i >= len(toks): continue
            op = toks[i].rstrip(";")
            # Skip the "decoded bytes" comments (purely hex, no letters).
            if not op or not op[0].isalpha(): continue
            ops.append(op)
    return ops

def _round_up(x:int, m:int) -> int: return ((x + m - 1) // m) * m

def _occupancy(arch:str, block_size:int, regs_per_thread:int, smem_per_block:int) -> dict:
    """Compute theoretical occupancy. Returns a dict with limits and active warps."""
    if arch not in SM_LIMITS: raise ValueError(f"unknown arch {arch!r}, want one of {list(SM_LIMITS)}")
    lim = SM_LIMITS[arch]
    warps_per_block = math.ceil(block_size / 32)
    # Register allocation: per-warp registers rounded to alloc granularity (256 for CC 7.x+).
    regs_per_warp_alloc = _round_up(regs_per_thread * 32, lim["warp_alloc"])
    blocks_by_reg = lim["regs"] // (regs_per_warp_alloc * warps_per_block) if regs_per_thread > 0 else lim["blocks"]
    # Shared memory: 128-byte allocation granularity is approximated as 1 here (NVCC pads it).
    blocks_by_smem = (lim["smem"] // smem_per_block) if smem_per_block > 0 else lim["blocks"]
    blocks_by_warp = lim["warps"] // warps_per_block
    blocks_by_hw = lim["blocks"]
    if warps_per_block > lim["warps"]: blocks_by_warp = 0  # block too big to fit
    active_blocks = min(blocks_by_reg, blocks_by_smem, blocks_by_warp, blocks_by_hw)
    active_warps = active_blocks * warps_per_block
    return dict(
        arch=arch, block_size=block_size, warps_per_block=warps_per_block,
        regs_per_thread=regs_per_thread, regs_per_warp_alloc=regs_per_warp_alloc,
        smem_per_block=smem_per_block,
        blocks_by_reg=blocks_by_reg, blocks_by_smem=blocks_by_smem,
        blocks_by_warp=blocks_by_warp, blocks_by_hw=blocks_by_hw,
        active_blocks=active_blocks, active_warps=active_warps,
        max_warps=lim["warps"], occupancy=active_warps/lim["warps"],
        limit_factor=min([(blocks_by_reg, "registers"), (blocks_by_smem, "shared mem"),
                          (blocks_by_warp, "warps/SM"), (blocks_by_hw, "hw blocks/SM")])[1],
    )

def _run_cuobjdump(cubin:bytes, flag:str) -> str:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "k.cubin").write_bytes(cubin)
        cmd = ["docker", "run", "--rm", "--entrypoint", "cuobjdump",
               "-v", f"{td}:/w", "-w", "/w", NVCC_IMAGE, flag, "k.cubin"]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr.decode(errors="replace"))
            raise SystemExit(f"cuobjdump {flag} failed (exit {r.returncode})")
        return r.stdout.decode(errors="replace")

def analyze(src: Path, entry: str|None, arch: str, block: tuple[int,int,int]|None,
            verbose: bool = False) -> None:
    cubin = compile_cu(src, arch, verbose)
    resource_text = _run_cuobjdump(cubin, "--dump-resource-usage")
    sass_text = _run_cuobjdump(cubin, "--dump-sass")
    usage = _parse_resource_usage(resource_text)
    if not usage: raise SystemExit("cuobjdump found no kernels in cubin")
    target = entry if entry in usage else (entry or next(iter(usage)))
    if target not in usage:
        raise SystemExit(f"entry {entry!r} not in cubin (found: {list(usage)})")
    u = usage[target]
    regs, smem = u.get("REG", 0), u.get("SHARED", 0)
    local, stack = u.get("LOCAL", 0), u.get("STACK", 0)
    cmem0 = u.get("CONSTANT[0]", 0)

    print(f"cumac analyze  {src.name}  entry={target}  arch={arch}")
    print()
    print("▸ resources")
    print(f"    registers/thread    {regs}")
    print(f"    static shared mem   {smem} B")
    print(f"    local mem/thread    {local} B")
    print(f"    stack/thread        {stack} B")
    print(f"    cmem[0] (params)    {cmem0} B")

    if block is not None:
        bs = block[0] * block[1] * block[2]
        print()
        print(f"▸ occupancy at block={bs} threads ({block[0]}×{block[1]}×{block[2]})")
        occ = _occupancy(arch, bs, regs, smem)
        print(f"    warps/block         {occ['warps_per_block']}")
        print(f"    regs/warp (alloc)   {occ['regs_per_warp_alloc']}")
        print(f"    blocks/SM by regs   {occ['blocks_by_reg']}")
        print(f"    blocks/SM by smem   {occ['blocks_by_smem']}")
        print(f"    blocks/SM by warps  {occ['blocks_by_warp']}")
        print(f"    blocks/SM hw limit  {occ['blocks_by_hw']}")
        print(f"    ─────")
        print(f"    active blocks/SM    {occ['active_blocks']}  (limit: {occ['limit_factor']})")
        print(f"    active warps/SM     {occ['active_warps']} / {occ['max_warps']}")
        print(f"    occupancy           {occ['occupancy']*100:.0f}%")

    ops = _parse_sass(sass_text, target)
    if not ops:
        print("\n(no SASS instructions parsed)"); return
    # Drop NOPs from the headline mix — they're padding, not work.
    real_ops = [op for op in ops if not op.startswith("NOP")]
    counts: dict[str, list[tuple[str,int]]] = {}
    op_counts: dict[str, int] = {}
    for op in real_ops:
        op_counts[op] = op_counts.get(op, 0) + 1
    for op, n in op_counts.items():
        counts.setdefault(_categorize_op(op), []).append((op, n))
    total = sum(op_counts.values())
    print()
    print(f"▸ instruction mix      ({total} instructions, {len(ops)-total} NOPs ignored)")
    for cat in ["mem", "fp", "int", "ctrl", "mov", "other"]:
        items = counts.get(cat, [])
        if not items: continue
        n = sum(c for _, c in items)
        items.sort(key=lambda x: -x[1])
        head = "  ".join(f"{c}× {op}" for op, c in items[:5])
        if len(items) > 5: head += f"  +{len(items)-5} more"
        print(f"    {cat:6s} {n:4d}  {n/total*100:3.0f}%   {head}")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        prog="cumac",
        description="Write a .cu kernel, run it on Mac via tinygrad's NV runtime.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile .cu → .cubin via Docker NVCC")
    c.add_argument("file", type=Path)
    c.add_argument("-a", "--arch", default="sm_86")  # 3090 = sm_86
    c.add_argument("-o", "--out", type=Path, default=None)
    c.add_argument("-v", "--verbose", action="store_true")

    r = sub.add_parser("run", help="compile (if needed) + launch on the eGPU")
    r.add_argument("file", type=Path)
    r.add_argument("entry", help="kernel entry symbol (use extern \"C\")")
    r.add_argument("args", nargs="*", help="kernel args in declaration order")
    r.add_argument("--grid",  required=True, help="Gx or Gx,Gy,Gz")
    r.add_argument("--block", required=True, help="Bx or Bx,By,Bz")
    r.add_argument("-a", "--arch", default="sm_86")
    r.add_argument("-v", "--verbose", action="store_true")

    z = sub.add_parser("analyze", help="static analysis: occupancy + SASS instruction mix")
    z.add_argument("file", type=Path)
    z.add_argument("entry", nargs="?", default=None, help="kernel entry symbol (default: first in cubin)")
    z.add_argument("--block", default=None, help="Bx[,By,Bz] — required to compute occupancy")
    z.add_argument("-a", "--arch", default="sm_86")
    z.add_argument("-v", "--verbose", action="store_true")

    a = p.parse_args()
    try:
        if a.cmd == "compile":
            cubin = compile_cu(a.file, a.arch, a.verbose)
            out = a.out or a.file.with_suffix(".cubin")
            out.write_bytes(cubin)
            print(f"wrote {len(cubin)} bytes → {out}")
        elif a.cmd == "run":
            cubin = compile_cu(a.file, a.arch, a.verbose)
            kargs = [parse_arg(s) for s in a.args]
            run_kernel(cubin, a.entry, kargs,
                       parse_dim(a.grid), parse_dim(a.block), a.verbose)
        elif a.cmd == "analyze":
            block = parse_dim(a.block) if a.block else None
            analyze(a.file, a.entry, a.arch, block, a.verbose)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr); return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
