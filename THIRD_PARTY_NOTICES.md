# Third-party notices

本仓库只收录构建两个渲染后端所需的最小第三方源码集合，并保留原始版权声明。

| 组件 | 仓库内位置 | 许可与来源 |
|---|---|---|
| Gaussian rasterizer forward | `jittor_hybridsplat/ops/native/vendor/standard_rasterizer` | Gaussian Splatting License，研究与评估用途，限制商业使用；完整文本随源码保存 |
| FlashGS deterministic forward | `jittor_hybridsplat/ops/native/vendor/flashgs` | MIT License，Copyright © 2024 InternLandMark |
| GLM | `jittor_hybridsplat/ops/native/vendor/standard_glm` 及 FlashGS 自带 `glm` | Happy Bunny License 或 MIT License；许可文本随源码保存 |
| OptiX tracing device code | `jittor_hybridsplat/ops/native/vendor/optix_tracer` | 来自 HybridSplat 修改的 diff-surfel-tracing/EnvGS 路径；保留文件内作者与 NVIDIA 声明 |
| NVIDIA OptiX SDK | 不随仓库分发 | 构建时由使用者提供 OptiX 7.7 SDK headers，并自行接受 NVIDIA SDK 许可 |

本仓库没有复制 PyTorch、ATen、c10、PyBind 或 OptiX SDK headers。若以后增加第三方源码，应先核对其再分发条件，并在同一提交中更新本文件和对应许可证。
