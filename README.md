# LLM Inference Power Simulation — GV100/V100-parameterised architectural model

Cycle-accurate power simulation of a small transformer model running LLM inference, using **GPGPU-Sim 4.2.0** and **AccelWattch** — no physical GPU required.

> **Simulation resolution: 0.691 ns per sample** (one simulated clock cycle at 1447 MHz — GPGPU-Sim config clock for GV100; real V100 SXM2 boost clock is 1530 MHz / 0.654 ns)

This is a **parameterised architectural model**, not a full hardware simulation. The GPU configuration (80 SMs, 6 MB L2, 32 GB HBM2) matches the V100 SXM2 specification. Execution runs in PTX functional simulation mode — Volta-specific features including tensor cores and independent thread scheduling are not modelled. See [Limitations](#limitations) for full details.

---

## Overview

This project builds a cycle-accurate architectural model of LLM inference on a GV100/V100-class GPU. Using GPGPU-Sim 4.2.0 with AccelWattch, it captures per-cycle power across all GPU components at 0.691 ns resolution — far beyond what nvidia-smi (1 ms) can observe.

The simulation captures power consumption across the prefill and decode phases of a small decoder-only transformer, enabling analysis of the compute-bound to memory-bound transition at a granularity that real hardware measurement tools cannot achieve.

**What this simulation models faithfully:**
- GV100/V100 microarchitectural parameters (SM count, cache hierarchy, clock, HBM)
- Per-cycle power across 34 components via AccelWattch
- Prefill vs decode phase structure and relative power behaviour
- L1/L2 cache dynamics and KV-cache access patterns
- Warp scheduling and shared memory utilisation

**What this simulation does not model:**
- Volta-specific ISA (PTX compiled for sm_52 — Maxwell PTX)
- Tensor core execution (wmma instructions not present)
- Real HBM bandwidth saturation (model is L2-resident at this size)
- Absolute power numbers matching real V100 hardware

---

## Model configuration

| Parameter | Value |
|---|---|
| Architecture | Transformer (decoder-only) |
| Hidden dimension | 128 |
| FFN dimension | 512 |
| Number of layers | 4 |
| Attention heads | 1 (implicit — no `NUM_HEADS` define; single-head behaviour) |
| Batch size | 8 |
| Prefill sequence length | 16 tokens |
| Decode steps | 16 tokens generated |
| Precision | FP32 |
| Model size | ~2 MB |

---

## GPU configuration

| Parameter | Value | Notes |
|---|---|---|
| GPU | NVIDIA V100 (GV100) | SXM2 variant (32 GB HBM2) |
| SMs | 80 | 5,120 CUDA cores total |
| Boost clock | 1530 MHz | SXM2 hardware spec; GPGPU-Sim config uses 1447 MHz |
| L1 / shared memory | 128 KB per SM | Unified pool; up to 96 KB configurable as shared memory, remainder as L1 data cache |
| L2 cache | 6 MB | 768 KB per memory controller × 8 |
| HBM2 | 32 GB @ 877 MHz | ~900 GB/s peak bandwidth |
| Simulator | GPGPU-Sim 4.2.0 | PTX functional simulation mode |
| Power model | AccelWattch | PTX sim mode; models per-cycle power across 34 components |
| Compilation target | sm_52 (Maxwell PTX) | GPGPU-Sim PTX mode constraint; Volta-specific instructions not modelled |

---

## Key findings

### Power profile

- Prefill average: 64.8 W
- Decode average: 58.4 W
- Dynamic power delta: ~6 W above a ~51 W structural baseline (CONSTP ~32 W + IDLE_COREP ~19 W)
- Peak power: 199.7 W — within V100 SXM2 TDP (300 W) ✓
- Total simulation cycles: 35,088

The prefill/decode averages should be read in context: approximately 51 W of the ~60 W average load is structural overhead (constant power + idle core power), present regardless of workload. The scientifically meaningful result is the relative dynamic power difference between phases, not the absolute wattage.

### Cache hierarchy

- The ~2 MB model (hidden=128, FFN=512, 4 layers, FP32) fits entirely in the 6 MB L2 cache after warm-up
- Zero DRAM reads after initial weight loading *(note: this is a consequence of the small model size, not a general result — production inference is HBM-bandwidth-bound because weights do not fit in L2)*
- L1 hit rate: 73.7% prefill GEMM vs 29.2% decode GEMM — quantifies the compute-bound to memory-bound transition *(the drop reflects reduced weight reuse per token in decode, not HBM pressure — the model remains L2-resident throughout)*

### Power components

- Constant power (CONSTP): ~32 W
- Idle core power (IDLE_COREP): ~19 W
- Structural baseline total: ~51 W (present regardless of workload phase)
- Shared memory power (SHRDP): 0.196 W — active, confirms tiled GEMM shared memory is exercised *(was 0 W in v1)*
- Integer division power (INT_DIVP): 0 W — LayerNorm modulo addressing eliminated *(was 0.010 W in v1)*

### Transformer architecture visible from power trace

Each prefill layer produces 8 distinct power regions in sequence:

`GEMM → attention_qk → softmax → attention_sv → GEMM → ReLU → GEMM → LayerNorm`

The three GEMMs correspond to QKV projection, output projection, and FFN respectively. With a single attention head at hidden=128, attention regions are narrow relative to the GEMM regions — multi-head configurations would show proportionally wider attention phases.

---

## Interactive visualisation

Explore the full per-cycle power trace with zoom, pan and component toggle:

**[Open interactive chart](https://dobutler.github.io/LLM-Inference-Simulation/Inference_v2_interactive.html)**

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
| `kv_cache_read` | Read K/V context — window grows with each decode step |

---

## Simulation infrastructure

```
CUDA source (inference_llm.cu)
        ↓
   nvcc -arch=sm_52  ← Maxwell PTX (GPGPU-Sim PTX mode constraint)
        ↓
      PTX bytecode
        ↓
GPGPU-Sim 4.2.0  ← GV100/V100-parameterised architectural model
(PTX functional simulation — Volta ISA not executed)
        ↓
AccelWattch  ← per-cycle power across 34 components
(PTX sim mode — tensor cores not modelled)
        ↓
Power trace + metric trace (per cycle, 0.691 ns resolution)
```

**Configuration files:**
- `gpgpusim.config` — GV100/V100 architecture parameters
- `accelwattch_ptx_sim.xml` — AccelWattch PTX simulation power model
- `config_volta_islip.icnt` — interconnect configuration

**Wall-clock simulation time:** 4 hours 40 minutes

---

## Limitations

**Simulation mode — PTX, not PTX+SASS**

GPGPU-Sim 4.x supports two simulation modes: PTX (functional) and SASS (microarchitectural). This project uses PTX mode, which is required for AccelWattch power modelling. PTX mode does not execute Volta-specific SASS instructions, meaning tensor cores and FP16 throughput are not modelled. All kernels run as FP32. Power numbers and cycle counts should be treated as indicative of relative behaviour rather than as predictions of real V100 hardware performance.

**Compilation target — sm_52 (Maxwell PTX)**

The CUDA kernel is compiled with `-arch=sm_52`, generating Maxwell-era PTX rather than Volta (`sm_70`). This is a constraint of GPGPU-Sim PTX mode. No Volta-specific instructions (independent thread scheduling, wmma tensor core ops) appear in the simulation trace. For v3, recompiling with `-arch=sm_70` and adding wmma intrinsics is planned.

**Model size and cache residency**

The ~2 MB model fits entirely in the 6 MB L2 cache after warm-up, eliminating DRAM traffic after the first pass. This is by design: the configuration isolates on-chip behaviour (L1/L2 cache dynamics, warp scheduling, register file pressure) without confounding from HBM bandwidth. Real production inference is HBM-bandwidth-bound because weights do not fit in on-chip caches. The L1 hit-rate drop from prefill to decode (73.7% → 29.2%) reflects reduced arithmetic intensity and growing KV-cache access patterns, not DRAM bandwidth saturation.

**Power baseline**

Of the ~60 W average load, approximately 51 W is structural overhead: CONSTP (~32 W) and IDLE_COREP (~19 W). Dynamic power — the component that varies meaningfully between prefill and decode phases — is a smaller fraction of the total. The 64.8 W / 58.4 W headline figures should be read in this context.

**Single attention head**

With one attention head and hidden dimension 128, attention kernels contribute a small share of total compute versus the FFN GEMMs. This differs from production models where multi-head attention and growing KV-cache dominate decode. The present configuration is appropriate for establishing a methodology baseline.

---

## Reproduce

**Environment**

| Dependency | Version |
|---|---|
| GPGPU-Sim | 4.2.0 (`dev` branch) |
| AccelWattch | bundled with GPGPU-Sim 4.2.0 |
| CUDA toolkit | 11.8 (host compiler only — nvcc produces PTX) |
| Compilation target | `-arch=sm_52` (PTX mode requirement) |
| OS | Ubuntu 22.04 |

**Steps**

```bash
# 1. Build GPGPU-Sim with AccelWattch
git clone https://github.com/accel-sim/gpgpu-sim_distribution
cd gpgpu-sim_distribution && git checkout dev
source setup_environment
make -j$(nproc)

# 2. Compile the inference kernel
nvcc -arch=sm_52 -O2 inference_llm.cu -o inference_llm

# 3. Run simulation (expect ~4h 40min wall-clock)
./inference_llm

# 4. Parse and visualise
python scripts/parse_trace.py
```

**Expected outputs**

| File | Description |
|---|---|
| `gpgpusim_power_trace_report_*.gz` | Per-cycle power for all 34 components |
| `gpgpusim_metric_trace_report_*.gz` | Per-cycle microarchitecture metrics |
| `accelwattch_power_report.log` | Per-kernel average power summary |
| `console.log` | Full GPGPU-Sim output with IPC and cache stats |

---

## References

[1] A. Bakhoda, G. L. Yuan, W. W. L. Fung, H. Wong, and T. M. Aamodt, "Analyzing CUDA workloads using a detailed GPU simulator," in *Proc. IEEE Int. Symp. Performance Analysis of Systems and Software (ISPASS)*, Apr. 2009, pp. 163–174.

[2] M. Khairy, J. Shen, T. M. Aamodt, and T. G. Rogers, "Accel-Sim: An extensible simulation framework for validated GPU modeling," in *Proc. 47th Int. Symp. Computer Architecture (ISCA)*, May 2020, pp. 473–486.

[3] V. Kandiah, S. Peverelle, M. Khairy, J. Pan, A. Manjunath, T. G. Rogers, T. M. Aamodt, and N. Hardavellas, "AccelWattch: A power modeling framework for modern GPUs," in *Proc. 54th IEEE/ACM Int. Symp. Microarchitecture (MICRO)*, Oct. 2021, pp. 738–753.

[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[5] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer normalization," *arXiv preprint arXiv:1607.06450*, 2016.

[6] P. J. Maliakel, S. Ilager, and I. Brandic, "Characterizing LLM inference energy-performance tradeoffs across workloads and GPU scaling," *arXiv preprint arXiv:2501.08219*, 2025.

[7] A. Solovyeva et al., "The illusion of power capping in LLM decode: A phase-aware energy characterisation across attention architectures," *arXiv preprint arXiv:2605.11999*, 2025.

---

