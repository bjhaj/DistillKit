import torch
import time
import os
import argparse
# Optional: MACs / FLOPs
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False

def measure_latency_throughput(model, input_shape=(1, 3, 32, 32), num_runs=100, device='cpu'):
    model.to(device)
    model.eval()
    torch.set_num_threads(1)  # Simulate edge conditions

    dummy_input = torch.randn(*input_shape).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure time
    with torch.no_grad():
        start = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
        end = time.time()

    total_time = end - start
    latency_ms = (total_time / num_runs) * 1000
    throughput = num_runs / total_time

    return latency_ms, throughput

def get_model_size(path):
    size_mb = os.path.getsize(path) / 1e6
    return round(size_mb, 2)

def measure_flops(model, input_shape=(1, 3, 32, 32)):
    if not FVCORE_AVAILABLE:
        return "fvcore not installed"
    input_tensor = torch.randn(*input_shape)
    flops = FlopCountAnalysis(model, input_tensor)
    return flop_count_table(flops, max_depth=2)