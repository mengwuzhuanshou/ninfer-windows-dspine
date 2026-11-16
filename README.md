---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - tencent/Qwen3.8-27B-DSpine-b16
pipeline_tag: text-generation
tags:
  - speculative-decoding
  - dspine
  - nvfp4
  - quantization
  - dflash2
  - windows
  - cuda
  - rtx5090
---

# NInfer Windows DSpine draft ｜ NInfer Windows 版 DSpine 草稿

A Windows / CUDA 13.1 port of the [NInfer](https://github.com/Neroued/ninfer) engine with a
from-scratch **DSpine** speculative-decode draft backend, plus the **DSpine draft model** in
BF16 and NVFP4. Tuned for a single RTX 5090 (32 GB), `sm_120a`.

[NInfer](https://github.com/Neroued/ninfer) 引擎的 Windows / CUDA 13.1 移植，自带从零实现的
**DSpine** 投机解码草稿后端，并附 **DSpine 草稿模型**的 BF16 与 NVFP4 两种精度。面向单张
RTX 5090（32 GB）、`sm_120a` 调优。

> The Qwen3.8-27B base tower and the Coder390 fine-tune tower are **not** redistributed here
> (Apache-2.0 attribution kept in `NOTICE`). Download them from their sources, then **fuse** the
> shipped DSpine draft suffix onto your tower (below). This repo ships the draft + the engine.

> Qwen3.8-27B 基塔与 Coder390 微调塔**不在本仓分发**（Apache-2.0 归因见 `NOTICE`）。请自原始来源
> 下载，再把本仓的 DSpine 草稿后缀**融合**到你的塔上（见下）。本仓只发草稿 + 引擎。

> **Download split ｜ 下载分布**：GitHub LFS caps a single file at 2 GiB, so the BF16 source
> checkpoint (3.77 GiB) and the two `-bf16tab` draft suffixes (2.17 GiB each) live **only on
> Hugging Face** ([mengwuzhuanshou/ninfer-windows-dspine](https://huggingface.co/mengwuzhuanshou/ninfer-windows-dspine)).
> GitHub carries the source, the exe, and both **NVFP4-table** draft suffixes (the adopted
> production form, each < 2 GiB). HF carries everything.
>
> GitHub LFS 单文件硬限 2 GiB，故 BF16 源 checkpoint（3.77 GiB）与两个 `-bf16tab` 草稿后缀
> （各 2.17 GiB）**只在 Hugging Face** 提供。GitHub 放源码、exe 与两个 **NVFP4 表**草稿后缀
> （生产采纳形态，各 < 2 GiB）；HF 放全部。

## What is DSpine ｜ 什么是 DSpine

[DSpine](https://github.com/tencent/dspine) (Tencent, Apache-2.0) is a speculative-decode draft
model: a 5-layer sliding-attention DFlash backbone (window 8192, block 16 → 15 draft tokens)
plus a selected-prefix **code bridge** and a **serial-refinement tail**. The draft proposes 15
tokens per step; the target tower verifies them. Acceptance is lossless w.r.t. draft precision —
the final text is the target's greedy output.

[DSpine](https://github.com/tencent/dspine)（腾讯，Apache-2.0）是投机解码草稿模型：5 层滑窗注意力
DFlash 骨干（窗口 8192、块 16 → 15 个草稿 token）+ 选定前缀**码桥**+**串行精修尾**。草稿每步提
15 个 token，由目标塔验证。接受对草稿精度无损——最终文本即目标塔的贪心输出。

## Repository layout ｜ 目录

| Path | Content |
|---|---|
| `src/` | Windows-port engine source (Nov 2026 upstream + our DSpine / NVFP4-table additions) |
| `bin/ninfer-serve-dspine.exe` | Prebuilt engine (Windows x64, CUDA 13.1, sm_120a) |
| `converter/` | Vendored conversion toolchain (for rebuilding the draft from the BF16 checkpoint) |
| `scripts/` | `fuse_dspine_draft.py`, `extract_dspine_draft.py`, `attach_dspine_head.py`, helpers |
| `launch/` | `serve-dspine-official.bat`, `serve-dspine-coder390.bat` (port 8080) |
| `draft/dspine-draft-official-bf16tab.ninfer` | DSpine draft suffix for the **official** tower, BF16 codebook/TG/TSV *(HF only)* |
| `draft/dspine-draft-official-nvfp4tab.ninfer` | DSpine draft suffix for the **official** tower, NVFP4 tables |
| `draft/dspine-draft-coder390-bf16tab.ninfer` | DSpine draft suffix for the **Coder390** tower, BF16 tables *(HF only)* |
| `draft/dspine-draft-coder390-nvfp4tab.ninfer` | DSpine draft suffix for the **Coder390** tower, NVFP4 tables |
| `draft/dspine-b16-bf16/` | Tencent DSpine-b16 BF16 source checkpoint (`model.safetensors` + `config.json`) *(HF only)* |

The draft suffix is **tower-specific**: the TG/TSV serial tables are computed against the target
tower's head, so use the `official` suffix with the official tower and the `coder390` suffix with
the Coder390 tower. The backbone and codebook are identical across towers.

草稿后缀**与塔绑定**：TG/TSV 串行表是按目标塔头算的，官方塔配 `official` 后缀、Coder390 塔配
`coder390` 后缀，不可混用。骨干与码桥两塔相同。

## Setup ｜ 部署

1. Obtain a target tower `.ninfer` (DFlash2 base, no DSpine head):
   - official: Qwen3.8-27B NVFP4 v3 DFlash2 artifact.
   - Coder390: Merkyor EfficientThink tokfix DFlash2 artifact.
   Place it under `models/` (e.g. `models/official-dflash2.ninfer`).
2. Fuse the draft suffix onto the tower (pure byte copy, no requantization):

   ```bat
   python scripts\fuse_dspine_draft.py models\official-dflash2.ninfer ^
       draft\dspine-draft-official-nvfp4tab.ninfer models\official.dspine.ninfer
   ```

   For Coder390 use `draft\dspine-draft-coder390-nvfp4tab.ninfer`. For BF16 tables use the
   `-bf16tab` suffix (larger draft, see VRAM below).
3. Run the launcher (default port 8080, `--model-id t`):

   ```bat
   launch\serve-dspine-official.bat
   ```

Only one engine can run at a time on the default port. The engine serves an OpenAI-compatible
`/v1/chat/completions`; speculative decode is enabled with `--spec dspine --draft-tokens 15`.

同一默认端口一次只能起一个引擎。引擎提供 OpenAI 兼容 `/v1/chat/completions`，投机解码由
`--spec dspine --draft-tokens 15` 开启。

### Rebuild the draft from the BF16 checkpoint (optional) ｜ 从 BF16 checkpoint 重建（可选）

```bat
rem BF16 tables:
python scripts\attach_dspine_head.py models\official-dflash2.ninfer draft\dspine-b16-bf16 models\official.dspine.ninfer
rem NVFP4 tables (saves ~869 MiB):
set NINFER_DSPINE_TABLES_NVFP4=1
python scripts\attach_dspine_head.py models\official-dflash2.ninfer draft\dspine-b16-bf16 models\official.dspine.ninfer
```

`attach_dspine_head.py` quantizes the backbone to NVFP4 and computes the tower-specific TG/TSV
tables; it uses the vendored `converter/` toolchain (PyTorch required). The shipped suffixes were
produced this way and verified byte-identical to a fused rebuild.

`attach_dspine_head.py` 把骨干量化为 NVFP4 并按塔头算 TG/TSV 表，使用 vendored `converter/`
工具链（需 PyTorch）。本仓后缀即由此产出，并已验证与融合重建逐字节一致。

## Measured: BF16 vs NVFP4 tables ｜ 实测：BF16 表 vs NVFP4 表

Same engine, same four prompts, temperature 0, official tower. Only the codebook/TG/TSV storage
format differs (backbone is NVFP4 in both):

同一引擎、同四题、temperature 0、官方塔。仅码桥/TG/TSV 存储格式不同（两臂骨干均 NVFP4）：

| Prompt | BF16 accept | NVFP4 accept | BF16 tok/s | NVFP4 tok/s | Output |
|---|---|---|---|---|---|
| counting | 100% | 100% | 850.3 | 858.4 | identical |
| code | 36.4% | 37.1% | 336.7 | 339.3 | 1-char near-tie flip |
| list | 18.9% | 18.6% | 200.5 | 198.8 | identical |
| prose | 20.3% | 20.3% | 211.3 | 211.3 | identical |

- **VRAM**: draft weights 2218.7 MiB (BF16 tables) → 1347.3 MiB (NVFP4 tables), **saves ~869 MiB**.
  Loaded engine `used` 25231 → 24362 MiB; post-load `free` 6.88 → 8.59 GiB.
- NVFP4 table quantization rel_err: codebook 0.0952, TG 0.0944, TSV 0.0941 (same order as backbone).
- The single 1-char code divergence is a near-tie greedy flip at a comment near the truncation
  boundary (both outputs are correct quicksort). Speculative decode is lossless w.r.t. draft
  precision; such flips come from target-tower batch-shape FP non-associativity, not table
  corruption. A control (BF16 tables, `--draft-tokens 14`, acceptance 38.9%) stayed byte-identical
  to BF16 K15, and the NVFP4 arm was byte-deterministic across reruns.

- **显存**：草稿权重 2218.7 MiB（BF16 表）→ 1347.3 MiB（NVFP4 表），**省约 869 MiB**。
- 那 1 字符差异是截断边界附近注释行的近 tie 贪心翻词（两段都是正确快排）；投机解码对草稿精度无损，
  翻词源自目标塔批形状 FP 非结合性，非查表污染。

## License ｜ 许可

Apache License 2.0 (see `LICENSE`). Third-party attribution in `NOTICE`: NInfer (Neroued),
the Windows fork (natpate/ninfer-windows), the converter (MirkoCovizzi/ninfer-rtx5090-mobile),
Tencent DSpine, Qwen3.8-27B, and the Merkyor Coder390 fine-tune — all Apache-2.0.