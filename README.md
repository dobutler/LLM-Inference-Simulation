# LLM Inference Power Simulation — GV100/V100-parameterised architectural model, #DilekAndHerTinyLLM

**A "wind tunnel" for LLM inference power consumption: Simulation of a small transformer running prefill and decode on a modelled NVIDIA V100 GPU, using GPGPU-Sim 4.2.0 + AccelWattch, no physical GPU required.**

[**▶ Interactive power profile**](https://dobutler.github.io/LLM-Inference-Simulation/Inference_v2_interactive.html)

---

## Overview

This project builds a cycle-level architectural model of LLM inference on a GV100/V100-class GPU. Using GPGPU-Sim 4.2.0 with AccelWattch, it captures power across GPU components, decomposing the prefill and decode phases of a small decoder-only transformer and exposing the compute-bound → memory-bound transition at a granularity real hardware tools cannot reach.

**What the simulation resolves:**
- GV100/V100 microarchitectural parameters (80 SMs, cache hierarchy, clocks, HBM)
- Per-component power via AccelWattch, sampled every 500 cycles + at every kernel boundary
- Prefill vs decode phase structure, per-token decode sawtooth, per-kernel power attribution
- L1/L2 cache dynamics and KV-cache access patterns
- Warp scheduling, register file pressure, and shared memory utilisation

**Scope boundaries:**
- PTX functional mode (sm_52): captures architectural dynamics
- L2-resident model at this size: isolates on-chip behaviour by design

---

## Model configuration

| Parameter | Value |
|---|---|
| Architecture | Transformer (decoder-only) |
| Hidden dimension | 128 |
| FFN dimension | 512 |
| Layers | 4 |
| Attention heads | 1 |
| Batch size | 8 |
| Prefill sequence length | 16 tokens |
| Decode steps | 16 tokens generated |
| Precision | FP32 |
| Model size | ~3.1 MB weights (786,432 params, FP32) + 131 KB KV cache |

The full run launches **1,200 kernels** (176 prefill, 1,024 decode) over **16.94 M core cycles** (11.71 ms).

## GPU configuration

| Parameter | Value | Notes |
|---|---|---|
| GPU | NVIDIA V100 (GV100) | Quadro/PCIe-class, 250 W TDP (config clock 1447 MHz; the 300 W SXM2 part boosts to 1530 MHz) |
| SMs | 80 | 5,120 CUDA cores total |
| Core clock | 1447 MHz (config) | config value, PCIe/Quadro-class part; SXM2 boosts to 1530 MHz |
| L1 / shared memory | 128 KB per SM | Unified pool; up to 96 KB configurable as shared |
| L2 cache | 6 MB | |
| HBM2 | 850 MHz (config) | SXM2 hardware spec is 877 MHz / ~900 GB/s |
| Simulator | GPGPU-Sim 4.2.0 | PTX functional simulation mode |
| Power model | AccelWattch | 34 power components in the trace, PTX sim mode |
| Compilation target | sm_52 (Maxwell PTX) | GPGPU-Sim PTX mode constraint |

---

## Kernel structure per layer

The inference engine implements 8 CUDA kernels:

| Kernel | Description |
|---|---|
| `gemm` | Tiled matrix multiply with shared memory (TILE=16) |
| `attention_qk` | Scaled dot-product Q×Kᵀ |
| `softmax_rows` | Numerically stable row-wise softmax |
| `attention_sv` | Weighted sum of V |
| `relu` | ReLU activation |
| `layernorm` | Layer normalisation over full hidden dim; one thread per row, so in decode only 8 threads are active chip-wide, which is why it measures 0.64 IPC at 1.6% occupancy |
| `kv_cache_write` | Store K/V for current token position |
| `kv_cache_read` | Read K/V context: window grows with each decode step |

---

## References

[1] A. Bakhoda, G. L. Yuan, W. W. L. Fung, H. Wong, and T. M. Aamodt, "Analyzing CUDA workloads using a detailed GPU simulator," in *Proc. IEEE ISPASS*, Apr. 2009, pp. 163–174.

[2] M. Khairy, J. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An extensible simulation framework for validated GPU modeling," in *Proc. 47th ISCA*, May 2020, pp. 473–486.

[3] V. Kandiah, S. Peverelle, M. Khairy, J. Pan, A. Manjunath, T. G. Rogers, T. M. Aamodt, and N. Hardavellas, "AccelWattch: A power modeling framework for modern GPUs," in *Proc. 54th IEEE/ACM MICRO*, Oct. 2021, pp. 738–753.

---

## Acknowledgements

Built on [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) (UBC) and [AccelWattch](https://accel-sim.github.io/accelwattch.html) (Northwestern / Purdue / UBC). GV100 configuration adapted from the Accel-Sim distribution. 

---

## A note on how this document was produced

This project was produced through several days of interactive work between me and an AI assistant. As much as it models GPU power, it doubled as a test of the capability and reliability of current AI tools. Questions, corrections, and audits welcome. 

#DilekAndHerTinyLLM
