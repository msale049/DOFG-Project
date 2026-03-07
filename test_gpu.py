#!/usr/bin/env python3
"""
Minimal GPU diagnostic for Anvil H100.
Run: python test_gpu.py

If OOM despite nvidia-smi showing 0 processes:
  - Try PyTorch cu124: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  - Driver 13.0 + PyTorch cu121 can have compatibility issues on H100
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Ensure we use GPU 0 (in case CUDA_VISIBLE_DEVICES is wrong)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

print("=" * 50)
print("GPU Diagnostic")
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)'))
print("=" * 50)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FAIL: CUDA not available")
    exit(1)

print("Device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("Total memory:", props.total_memory / 1e9, "GB")

# Test 1: small allocation (run FIRST — if GPU is full, this will fail)
print("\nTest 1: 4MB allocation...")
try:
    x = torch.zeros(1000, 1000, device='cuda')
    print("  OK")
    del x
    torch.cuda.empty_cache()
except RuntimeError as e:
    print("  FAIL:", e)
    exit(1)

# Test 2: 100MB allocation
print("Test 2: 100MB allocation...")
try:
    x = torch.zeros(25000, 1000, device='cuda')  # 100MB
    print("  OK")
    del x
    torch.cuda.empty_cache()
except RuntimeError as e:
    print("  FAIL:", e)
    exit(1)

# Test 3: ResNet34-sized (~85MB)
print("Test 3: ResNet34-sized model (~85MB)...")
try:
    import torchvision.models as M
    m = M.resnet34(weights=None)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    m = m.cuda()
    print("  OK")
    del m
    torch.cuda.empty_cache()
except RuntimeError as e:
    print("  FAIL:", e)
    exit(1)

print("\n" + "=" * 50)
print("All tests passed. GPU is OK for pipeline.")
print("=" * 50)
