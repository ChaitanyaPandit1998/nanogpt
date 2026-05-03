"""
debug_compile.py
~~~~~~~~~~~~~~~~
Diagnose why torch.compile is not providing speedup.
Run this after stopping training at a checkpoint.

Usage:
    python debug_compile.py
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"Device   : {torch.cuda.get_device_name(0)}")
print(f"SM       : {torch.cuda.get_device_capability(0)}")
try:
    import triton
    print(f"Triton   : {triton.__version__}")
except ImportError:
    print("Triton   : NOT INSTALLED")

print()

# ---------------------------------------------------------------------------
# Test 1: simple matmul — does compile help at all?

print("=" * 60)
print("TEST 1 — Simple matmul speedup")
print("=" * 60)

B, T, C = 16, 2048, 768
x = torch.randn(B * T, C, device="cuda", dtype=torch.bfloat16)
w = torch.randn(C, C, device="cuda", dtype=torch.bfloat16)

def matmul_fn(x, w):
    return x @ w

compiled_matmul = torch.compile(matmul_fn, dynamic=False)

# Warmup
for _ in range(5):
    _ = matmul_fn(x, w)
    _ = compiled_matmul(x, w)
torch.cuda.synchronize()

# Benchmark eager
N = 100
t0 = time.perf_counter()
for _ in range(N):
    _ = matmul_fn(x, w)
torch.cuda.synchronize()
eager_ms = (time.perf_counter() - t0) / N * 1000

# Benchmark compiled
t0 = time.perf_counter()
for _ in range(N):
    _ = compiled_matmul(x, w)
torch.cuda.synchronize()
compiled_ms = (time.perf_counter() - t0) / N * 1000

print(f"  Eager    : {eager_ms:.2f} ms")
print(f"  Compiled : {compiled_ms:.2f} ms")
print(f"  Speedup  : {eager_ms / compiled_ms:.2f}×")
if compiled_ms < eager_ms * 0.9:
    print("  ✅ torch.compile IS working on this pod")
else:
    print("  ❌ torch.compile provides NO speedup — likely version issue")

print()

# ---------------------------------------------------------------------------
# Test 2: check for graph breaks in a GPT-like block

print("=" * 60)
print("TEST 2 — Graph break detection on attention block")
print("=" * 60)

import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.verbose = True   # prints graph breaks to stdout

class SimpleAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.out = nn.Linear(C, C, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=-1)
        q, k, v = [t.view(B, T, 12, C // 12).transpose(1, 2) for t in qkv]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out(attn.transpose(1, 2).contiguous().view(B, T, C))

model = SimpleAttn().cuda().bfloat16()
compiled_model = torch.compile(model, dynamic=False)

x = torch.randn(B, T, C, device="cuda", dtype=torch.bfloat16)
print("  Running compiled attention forward (graph breaks will print above)...")
with torch.no_grad():
    _ = compiled_model(x)
print("  Done — if no graph break messages appeared above, compilation is clean.")

print()

# ---------------------------------------------------------------------------
# Test 3: dynamo explain (shows graph break count)

print("=" * 60)
print("TEST 3 — torch._dynamo.explain()")
print("=" * 60)

torch._dynamo.reset()
torch._dynamo.config.verbose = False

try:
    explanation = torch._dynamo.explain(model)(x)
    print(f"  Graph count      : {explanation.graph_count}")
    print(f"  Graph break count: {explanation.graph_break_count}")
    if explanation.graph_break_count == 0:
        print("  ✅ No graph breaks — compile should work")
    else:
        print(f"  ❌ {explanation.graph_break_count} graph break(s) found — causing eager fallback")
        for reason in explanation.break_reasons[:5]:
            print(f"     - {reason}")
except Exception as e:
    print(f"  Could not run explain: {e}")

print()
print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)
if compiled_ms >= eager_ms * 0.9:
    print("  torch.compile is not working on this pod.")
    print("  Options:")
    print("  1. Set use_compile=False in train_gpt.py (accept slower speed)")
    print("  2. Try: pip install --upgrade torch triton")
    print("  3. Try mode='reduce-overhead' instead of dynamic=False")
else:
    print("  torch.compile works. The issue may be DDP+compile interaction.")
    print("  Try adding: torch._dynamo.config.optimize_ddp = False")
    print("  in train_gpt.py before torch.compile() call.")
print("=" * 60)
