# V100 LLM Inference Power Simulation

Cycle-accurate GPU power simulation of LLM inference on a simulated NVIDIA V100 (GV100) using **GPGPU-Sim 4.x** and **AccelWattch**.

---

## Overview

This project simulates a small transformer model running LLM inference on a V100 GPU without access to real hardware. The simulation captures per-cycle power consumption across all GPU components, enabling analysis at a granularity that real hardware measurement tools like nvidia-smi (1ms resolution) cannot achieve.

**Simulation resolution: 0.691 ns per sample** (one GPU clock cycle at 1447 MHz)

---

## Model configuration

| Parameter | Value |
|---|---|
| Architecture | Transformer (decoder-only) |
| Hidden dimension | 128 |
| FFN dimension | 512 |
| Number of layers | 4 |
| Attention heads | 1 |
| Batch size | 8 |
| Prefill sequence length | 16 tokens |
| Decode steps | 16 tokens generated |
| Precision | FP32 |

---

## GPU configuration

| Parameter | Value |
|---|---|
| GPU | NVIDIA V100 (GV100) |
| SMs | 80 |
| Clock | 1447 MHz |
| L1 cache | 128 KB per SM |
| L2 cache | 6 MB |
| HBM | 32 GB @ 850 MHz |
| Simulator | GPGPU-Sim 4.x |
| Power model | AccelWattch (PTX sim mode) |

---

## Key findings

**Power profile**
- Prefill average: 64.8 W
- Decode average: 58.4 W 
- Peak power: 199.7 W 
- Total simulation cycles: 35,088

**Cache hierarchy**
- Entire 2 MB model fits in the 6 MB L2 cache after warm-up
- Zero DRAM reads after initial weight loading
- L1 hit rate: 73.7% prefill GEMM vs 29.2% decode GEMM — quantifies the compute-bound to memory-bound transition

**Power components**
- Constant power (CONSTP): ~32 W 
- Idle core power (IDLE_COREP): ~19 W 
- Shared memory power (SHRDP): active 
- Integer division power (INT_DIVP): 0 W 

**Transformer architecture visible from power trace**
Each prefill layer produces 8 distinct power regions in sequence:
GEMM → attention_qk → softmax → attention_sv → GEMM → ReLU → GEMM → LayerNorm

---

## Interactive visualisation

Explore the full per-cycle power trace with zoom, pan and component toggle:

**[Open interactive chart](https://dobutler.github.io/V100-LLM-Inference-Simulation/v100_v2_interactive.html)**

Features:
- Box zoom on both X and Y axes
- Range slider for full trace navigation
- Toggle individual power components on/off
- Hover for exact per-cycle values
- Export current view as PNG

---

## CUDA kernels

The inference engine (`inference_llm.cu`) implements 8 CUDA kernels:

| Kernel | Description |
|---|---|
| `gemm` | Tiled matrix multiply with shared memory (TILE=16) |
| `attention_qk` | Scaled dot-product Q×Kᵀ |
| `softmax_rows` | Numerically stable row-wise softmax |
| `attention_sv` | Weighted sum of V |
| `relu` | ReLU activation |
| `layernorm` | Layer normalisation over full hidden dim |
| `kv_cache_write` | Store K/V for current token position |
| `kv_cache_read` | Read K/V context (grows with each decode step) |

---

## Simulation infrastructure

```
CUDA source (inference_llm.cu)
        ↓
   nvcc -arch=sm_52
        ↓
GPGPU-Sim (cycle-accurate V100 model)
        ↓
AccelWattch (per-cycle power model)
        ↓
Power trace + metric trace (per cycle)
```

**Configuration files:**
- `gpgpusim.config` — V100 architecture parameters
- `accelwattch_ptx_sim.xml` — AccelWattch PTX simulation power model
- `config_volta_islip.icnt` — interconnect configuration

**Wall-clock simulation time:** 4 hours 40 minutes

---

## Output files

| File | Description |
|---|---|
| `gpgpusim_power_trace_report_*.gz` | Per-cycle power for all 34 components |
| `gpgpusim_metric_trace_report_*.gz` | Per-cycle microarchitecture metrics |
| `accelwattch_power_report.log` | Per-kernel average power summary |
| `console.log` | Full GPGPU-Sim output with IPC and cache stats |

---

## References

[1] A. Bakhoda, G. L. Yuan, W. W. L. Fung, H. Wong, and T. M. Aamodt,
    "Analyzing CUDA workloads using a detailed GPU simulator," in *Proc.
    IEEE Int. Symp. Performance Analysis of Systems and Software (ISPASS)*,
    Apr. 2009, pp. 163–174.

[2] M. Khairy, J. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An
    extensible simulation framework for validated GPU modeling," in *Proc.
    47th Int. Symp. Computer Architecture (ISCA)*, May 2020, pp. 473–486.

[3] V. Kandiah, S. Peverelle, M. Khairy, J. Pan, A. Manjunath, T. G. Rogers,
    T. M. Aamodt, and N. Hardavellas, "AccelWattch: A power modeling
    framework for modern GPUs," in *Proc. 54th IEEE/ACM Int. Symp.
    Microarchitecture (MICRO)*, Oct. 2021, pp. 738–753.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances
    in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[5] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization,"
    *arXiv preprint arXiv:1607.06450*, 2016.

[6] P. J. Maliakel, S. Ilager, and I. Brandic, "Characterizing LLM inference
    energy-performance tradeoffs across workloads and GPU scaling,"
    *arXiv preprint arXiv:2501.08219*, 2025.

[7] A. Solovyeva et al., "The illusion of power capping in LLM decode:
    A phase-aware energy characterisation across attention architectures,"
    *arXiv preprint arXiv:2605.11999*, 2025.


---
