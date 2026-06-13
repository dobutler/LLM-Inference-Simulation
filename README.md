# LLM Inference Power Simulation — GV100/V100-parameterised architectural model

**A "wind tunnel" for LLM inference energy: cycle-level simulation of a small transformer running prefill and decode on a modelled NVIDIA V100 GPU, using GPGPU-Sim 4.2.0 + AccelWattch — no physical GPU required.**

[**▶ Interactive power profile**](https://dobutler.github.io/LLM-Inference-Simulation/Inference_v2_interactive.html)

> **Trace resolution: ~345.5 ns per sample.** AccelWattch computes power every cycle internally and records one sample per 500 simulated cycles, plus one sample at every kernel completion (33,888 + 1,200 = 35,088 samples over 16,944,217 cycles = 11.71 ms at the 1447 MHz config clock). That is roughly **three orders of magnitude finer than millisecond-scale hardware tools such as nvidia-smi**, fine enough to resolve individual kernels and per-token structure that hardware measurement averages away.

---

## Overview

This project builds a cycle-level architectural model of LLM inference on a GV100/V100-class GPU. Using GPGPU-Sim 4.2.0 with AccelWattch, it captures power across 34 GPU components at ~345 ns trace resolution (vs millisecond-scale for hardware tools), decomposing the prefill and decode phases of a small decoder-only transformer and exposing the compute-bound → memory-bound transition at a granularity real hardware tools cannot reach.

**What the simulation resolves:**
- GV100/V100 microarchitectural parameters (80 SMs, cache hierarchy, clocks, HBM)
- Per-component power via AccelWattch, sampled every 500 cycles + at every kernel boundary
- Prefill vs decode phase structure, per-token decode sawtooth, per-kernel power attribution
- L1/L2 cache dynamics and KV-cache access patterns
- Warp scheduling, register file pressure, and shared memory utilisation

**Scope boundaries:**
- PTX functional mode (sm_52): captures architectural dynamics; Volta ISA and tensor cores are v3 targets
- L2-resident model at this size: isolates on-chip behaviour by design; HBM bandwidth scaling is v3 scope

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
| Core clock | 1447 MHz (config) | matches PCIe/Quadro class; SXM2 boosts to 1530 MHz |
| L1 / shared memory | 128 KB per SM | Unified pool; up to 96 KB configurable as shared |
| L2 cache | 6 MB | |
| HBM2 | 850 MHz (config) | SXM2 hardware spec is 877 MHz / ~900 GB/s |
| Simulator | GPGPU-Sim 4.2.0 | PTX functional simulation mode |
| Power model | AccelWattch | 34 components, PTX sim mode |
| Compilation target | sm_52 (Maxwell PTX) | GPGPU-Sim PTX mode constraint |

---

## Kernel structure per layer

The inference engine (`inference_llm.cu`) implements 8 CUDA kernels:

| Kernel | Description |
|---|---|
| `gemm` | Tiled matrix multiply with shared memory (TILE=16) |
| `attention_qk` | Scaled dot-product Q×Kᵀ |
| `softmax_rows` | Numerically stable row-wise softmax |
| `attention_sv` | Weighted sum of V |
| `relu` | ReLU activation |
| `layernorm` | Layer normalisation over full hidden dim; one thread per row — in decode only 8 threads are active chip-wide, which is why it measures 0.64 IPC at 1.6% occupancy |
| `kv_cache_write` | Store K/V for current token position |
| `kv_cache_read` | Read K/V context — window grows with each decode step |

---

## Interactive visualisation

**[Open interactive chart](https://dobutler.github.io/LLM-Inference-Simulation/Inference_v2_interactive.html)** — box zoom (X+Y), range slider, per-component toggle, hover for exact values, PNG export. The x-axis uses the exact reconstructed time base (`analysis/time_axis_ms.json`); phase and token boundaries are in `analysis/axis_metadata.json`.

---

## Simulation infrastructure

```
CUDA source (inference_llm.cu)
        ↓
   nvcc -arch=sm_52            ← Maxwell PTX (GPGPU-Sim PTX mode constraint)
        ↓
      PTX bytecode
        ↓
GPGPU-Sim 4.2.0                ← GV100/V100-parameterised architectural model
(PTX functional simulation — Volta ISA not executed)
        ↓
AccelWattch                    ← power across 34 components
(PTX sim mode — tensor cores not modelled)
        ↓
Power trace (1 sample / 500 cycles + 1 per kernel end ≈ 345.5 ns resolution)
```
---

## References

[1] A. Bakhoda, G. L. Yuan, W. W. L. Fung, H. Wong, and T. M. Aamodt, "Analyzing CUDA workloads using a detailed GPU simulator," in *Proc. IEEE ISPASS*, Apr. 2009, pp. 163–174.

[2] M. Khairy, J. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An extensible simulation framework for validated GPU modeling," in *Proc. 47th ISCA*, May 2020, pp. 473–486.

[3] V. Kandiah, S. Peverelle, M. Khairy, J. Pan, A. Manjunath, T. G. Rogers, T. M. Aamodt, and N. Hardavellas, "AccelWattch: A power modeling framework for modern GPUs," in *Proc. 54th IEEE/ACM MICRO*, Oct. 2021, pp. 738–753.

[4] A. Vaswani et al., "Attention is all you need," in *NeurIPS*, vol. 30, 2017.

[5] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv:1607.06450*, 2016.

[6] P. J. Maliakel, S. Ilager, and I. Brandic, "Characterizing LLM inference energy-performance tradeoffs across workloads and GPU scaling," *arXiv:2501.08219*, 2025.

[7] A. Solovyeva et al., "The illusion of power capping in LLM decode: A phase-aware energy characterisation across attention architectures," *arXiv:2605.11999*, 2026.

---

## Acknowledgements

Built on [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) (UBC) and [AccelWattch](https://accel-sim.github.io/accelwattch.html) (Northwestern / Purdue / UBC). GV100 configuration adapted from the Accel-Sim distribution. 

---

## A note on how this document was produced

This README and the underlying analysis were produced through several days of interactive work between the author and an AI assistant.
