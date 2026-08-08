# HybridSplat-diff-surfel-tracing

> **Modified version for [HybridSplat](https://github.com/Aetheryne/HybridSplat).**
> Original: [xbillowy/diff-surfel-tracing](https://github.com/xbillowy/diff-surfel-tracing), developed for [EnvGS](https://github.com/zju3dv/EnvGS) (CVPR 2025).

This is the differentiable OptiX-based 2D Gaussian ray tracer used by HybridSplat for **reflection-baked Gaussian tracing**. It traces per-Gaussian reflected rays and accumulates the incoming radiance as the `r_REF` reflection attribute stored on each reflective Gaussian.

## Modifications

### 1. `scores` parameter added to `SurfelTracer.forward()`

A `scores` optional argument (default `None`) is accepted by `SurfelTracer.forward()` and the underlying `trace_surfels()` function for API symmetry with the rasterizer. The gradient through `scores` is set to `None` in the backward pass — it is not used in the tracing backward, only surfaced for interface consistency.

### 2. `AUX_CHANNELS 2` in `config.h` (specular + roughness)

The `config.h` defines:

```c
#define AUX_CHANNELS 2  // 1 specular + 1 roughness
#define MID_CHANNELS 16  // 3 ray_o + 3 ray_d + 1 dpt + 1 acc + 3 normal + 2 aux + 3 RGB
```

The two auxiliary output channels (`out_aux`) carry **per-pixel specular** and **roughness** values from the traced reflective Gaussians. In HybridSplat, the specular channel is used to retrieve the baked reflection color `r_REF` for each reflective Gaussian after a single trace pass.

### 3. Per-Gaussian accumulated weights output (`acc_wet`)

The tracer returns `acc_wet` — the accumulated alpha-weighted hit count per Gaussian over the image — as its 8th output. HybridSplat uses this as part of the importance scoring in `grad_prune`:

```python
rgb, dpt, acc, norm, dist, aux, mid, acc_wet = tracer(...)
```

## Installation

Requires the NVIDIA OptiX SDK header files (included in `third_party/optix/`).

```bash
pip install -v .
```

For a custom OptiX installation, set `OPTIX_HOME` before building:

```bash
export OPTIX_HOME=/path/to/optix
pip install -v .
```

## Citation

If you use this tracer, please cite:

```bibtex
@article{xie2024envgs,
    title={EnvGS: Modeling View-Dependent Appearance with Environment Gaussian},
    author={Xie, Tao and Chen, Xi and Xu, Zhen and Xie, Yiman and Jin, Yudong and Shen, Yujun and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
    journal={arXiv preprint arXiv:2412.15215},
    year={2024},
}
```
