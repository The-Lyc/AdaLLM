# RunKV 历史测试与启动说明

整理日期：2026-10-21。范围是本项目新增的 RunKV 实验与回归测试，不展开上游 vLLM 的全部通用测试。

## 1. 证据与测试状态

本记录对照了 `/home/lyc/.zsh_history`、Git 提交历史、当前工作区脚本和配置，以及 `exp_results/`、`analysis_artifacts/` 中的 manifest、运行日志和报告。本次整理没有重新执行实验。

| 状态 | 判定方式 |
| --- | --- |
| 已完成 | 找到完成日志、成功退出码、完整分析结果或微基准汇总；仅针对明确列出的模型和批次 |
| 有启动记录 | history 中有命令，但不足以单独证明成功；失败或中断的尝试不计为完成 |
| 仅确认入口 | 脚本或配置存在，本次没有找到对应的完成证据；不表示断言从未运行 |

历史实际参数优先读取对应 manifest，其次是日志。**当前 JSON 和脚本默认值不等于历史运行参数，用例名也不一定准确。**文中的启动示例若没有注明“history 原命令”，均为当前入口的复现方式；使用新的 `RUN_TAG`，不复用历史结果标签。

| 测试类别 | 已完成范围与日期 | 仅确认入口或证据限制 |
| --- | --- | --- |
| TightLLM 离线 profiling | 2026-07-14/15/29；OPT-1.3B、2.7B、6.7B、Llama-2-7B 有 profile 文件 | 旧 Llama 小写 profile 不适用于当前 Llama planner |
| 常规 RunKV / TightLLM 对比 | 2026-07-14—08-06；OPT 系列和 Llama-2-7B | 不同日期负载和内存配置发生过调整 |
| 分阶段 IO 竞争 | OPT-1.3B/2.7B/6.7B；Llama-2-7B | 早期部分尝试失败，不能把每条 history 都算作完成 |
| 分阶段 SM 竞争 | OPT-2.7B、6.7B 有完成产物 | Llama SM 仅确认配置和入口 |
| 同卡双推理任务 | 2026-07-16；RunKV+RunKV、TightLLM+TightLLM | 以完整的 `case_summary.json` 为准 |
| RunKV + Flux | 2026-07-23；OPT-2.7B combined nsys 有成功退出码 | Flux batch 有启动记录；Llama Flux 未找到完成产物 |
| 组件消融 | 2026-08-10；OPT-2.7B，稳定/IO 环境各 4 负载 × 5 变体 | 最终批次有效，早先批次存在退出码 137 |
| Spec ON/OFF | 2026-08-31 补测，结合已有 OPT-2.7B 记录；三模型、两环境、四种负载有配对分析 | 原始启动命令未见于 history，示例由 manifest 重建 |
| segment-DMA 微基准 | 2026-08-27 有启动记录；2026-08-30 三模型有完整测量 | 两天的迭代次数和采样窗口不同 |
| 其他微基准、旧 vanilla 套件、正确性回归 | 见第 4 节 | 仅确认入口，不能宣称整套已通过 |

## 2. 公共环境与参数

下面命令使用本机绝对仓库路径，通过变量减少重复：

```bash
ROOT=/home/lyc/inference/vllm-RunKV
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=0
RUN_TAG="reproduce_$(date +%Y%m%d_%H%M%S)"
```

环境安装与 CUDA 编译步骤见 [README](../README.md)。本机 `exp_results` 是指向 `/data/lyc/exp_results` 的符号链接；下文的 `exp_results/...` 都以仓库根目录为基准。`analysis_artifacts/...` 位于仓库目录内。

7 月留存的 Nsight SQLite 元数据确认使用过 RTX 4090、AMD Ryzen 9 9950X 和 Ubuntu 22.04.5；8 月 30 日 DMA 报告记录 RTX 4090 24 GiB、PyTorch `2.11.0+cu128`、CUDA 12.8。这些信息只描述已核实批次，不保证所有历史运行使用相同软件环境。

| JSON 字段 / CLI | observation 环境变量 | 含义 |
| --- | --- | --- |
| `batch_size` / `--num-prompts` | `NUM_PROMPTS` | 总请求数；并发还受 `max_num_seqs` 和容量限制 |
| `prompt_length` / `--prompt-words` | `PROMPT_WORDS` | 重复构造词数量，不是精确 tokenizer token 数 |
| `decode_length` / `--max-tokens` | `MAX_TOKENS` | 生成 token 数；benchmark 通常固定输出长度 |
| `prefix_blocks` / `--prefix-blocks` | `PREFIX_BLOCKS` | 初始每层 IO prefix 的 block 数，不是回放 token 数 |
| `gpu_memory_utilization` | `GPU_MEMORY_UTILIZATION` | 引擎的 GPU 内存使用比例 |
| `gpu_memory_fraction` | `GPU_MEMORY_FRACTION` | GPU staging buffer 容量分配比例 |
| `num_device_buffers` | `NUM_DEVICE_BUFFERS` | GPU 环形 staging buffer 数 |
| `max_staging_blocks` | `MAX_STAGING_BLOCKS` | 每个 staging buffer 的显式 block 上限；未指定时自动计算 |
| `cpu_memory_gb` | `CPU_MEMORY_GB` | CPU backing-store 总预算，单位 GiB；动态回放时包含 KV 和 hidden states |
| `runkv_replay_allocation_policy` | `REPLAY_ALLOCATION_POLICY` | `spread` 或 `concentrate` 的请求间回放分配策略 |

正常测试主要通过 JSON → batch → pipeline → observation runner → `opt_replay_component_mfu.py` 执行。直接调用最底层脚本时，默认参数与外层不同，应显式传参。

## 3. 有完成证据的测试

### 3.1 TightLLM 离线 profiling

用途：测量模型 attention/FFN 效率、PCIe H2D 带宽和硬件参数，生成 TightLLM planner 输入。profile 与模型、GPU、PCIe 拓扑和软件栈有关。

OPT 启动方式，history 中也记录过同类命令：

```bash
python -m vllm.v1.profiling.tightllm_offline_profiler \
  --model /data/models/opt-2.7b-8k \
  --output "$ROOT/exp_results/tightllm_profiles/ubuntu/opt-2.7b-8k.json" \
  --seq-lengths 128 256 512 1024 2048 4096 8192 16384 \
  --device cuda:0
```

OPT-1.3B、6.7B 分别替换模型目录与输出文件名。Llama 的当前有效命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m vllm.v1.profiling.tightllm_offline_profiler \
  --model /data/models/Llama-2-7b-hf-8k \
  --output "$ROOT/exp_results/tightllm_profiles/ubuntu/Llama-2-7b-hf-8k.json" \
  --seq-lengths 128 256 512 1024 2048 4096 8192 \
  --device cuda:0
```

已有结果位于 `exp_results/tightllm_profiles/ubuntu/`：

- `opt-1.3b-8k.json`
- `opt-2.7b-8k.json`
- `opt-6.7b-8k.json`
- `Llama-2-7b-hf-8k.json`：当前 Llama benchmark 使用的 profile。
- `llama2-7b-8k.json`：旧文件，缺少 Llama 架构信息，不能替代上面的文件。

模型路径分别为 `/data/models/opt-{1.3b,2.7b,6.7b}-8k` 和 `/data/models/Llama-2-7b-hf-8k`。更多解释见 [Llama profiling 说明](runkv_llama2.md)。

### 3.2 常规 RunKV / TightLLM 对比

已完成范围：7 月 OPT-1.3B/2.7B/6.7B；7 月 29 日增加 Llama-2-7B。比较 feedback planner 与 TightLLM planner，收集 component timing、prehook timing、Nsight timeline 和逐层分析。

当前批量入口：

```bash
python "$ROOT/scripts/run_benchmark_batch.py" \
  "$ROOT/configs/benchmark_batch_opt_2.7b.json"

python "$ROOT/scripts/run_benchmark_batch.py" \
  "$ROOT/configs/benchmark_batch_opt_6.7b.json"

python "$ROOT/scripts/run_llama_benchmark_batch.py" \
  "$ROOT/configs/benchmark_batch_llama2_7b.json"
```

这些入口可以加 `--dry-run` 展开命令。OPT batch 即使 dry-run 也检查 profile 文件是否存在；Llama batch 可在 profile 尚未生成时展开。普通 batch 没有 `--test-index` 或 `--only` 参数。

当前配置的四组负载如下，`B/P/D` 表示请求数 / prompt words / 生成 tokens：

| 配置 | 1k：B/P/D | 2k：B/P/D | 4k：B/P/D | 8k：B/P/D |
| --- | --- | --- | --- | --- |
| [OPT-2.7B](../configs/benchmark_batch_opt_2.7b.json) | 32/1000/64 | 32/2000/64 | 32/4000/32 | 16/8000/32 |
| [OPT-6.7B](../configs/benchmark_batch_opt_6.7b.json) | 32/1000/64 | 32/2000/32 | 32/4000/32 | 16/8000/32 |
| [Llama-2-7B](../configs/benchmark_batch_llama2_7b.json) | 32/1000/64 | 32/2000/32 | 16/4000/32 | 8/8000/32 |

共同参数：CPU `47.9 GiB`、CPU fraction `0.8`、GPU staging fraction `0.9`、3 buffers、H2D `segment`；`max_num_seqs`、`max_staging_blocks` 未显式限制。

| 配置 | GPU utilization | 初始 prefix blocks | RunKV allocation | TightLLM allocation |
| --- | --- | --- | --- | --- |
| OPT-2.7B | 0.7 | 10000000 | concentrate | spread |
| OPT-6.7B | 0.9 | 10000000 | concentrate | concentrate |
| Llama-2-7B | 0.7 | 32、64、128、256 | concentrate | spread |

RunKV 使用 `feedback`、`DRY_RUN=0`、状态机开启、异步计划开启；普通 paired benchmark 默认采集 nsys/NVTX、组件与 prehook 计时，并导出 SQLite、执行分析。Llama 使用 `prompt_word=the`；OPT 使用 `replay`。

**历史实际值示例**：7 月 27 日 OPT-2.7B 1k 的 manifest 是 `B32/P1000/D128`、GPU utilization `0.6`、GPU staging fraction `0.9`、CPU `47.9 GiB`、prefix `10000000`。它与当前 JSON 的 `D64/GU0.7` 不同。对应标签为：

```text
20260727_153835_j001_opt2p7b-bs32-p1k-d64_opt-2p7b-8k_cpu47p9_gu0p6_gf0p9_rkcopy-segment_rkalloc-concentrate_tlcopy-segment_tlalloc-spread_bs32_p1000_d128
```

下面是目录规则，以完整标签替换 `<tag>`。该 7 月 27 日批次已确认存在两系统 manifest 和 SQLite；不能据此假定同名分析目录也一定完整：

| 产物 | OPT 路径 |
| --- | --- |
| 运行参数 | `exp_results/manifests/{runkv,tightllm}_<tag>.json` |
| component JSONL、Nsight | `exp_results/{opt_feedback_observation,tightllm_observation}/`，文件名含 tag |
| SQLite | `exp_results/sqlite/{runkv,tightllm}_<tag>.sqlite` |
| 逐层分析 | `exp_results/analysis/per_layer/<tag>/` |
| batch 日志 | `exp_results/logs/benchmark_batch/<tag>/pipeline.log` |

已核实存在的早期逐层完成报告示例是 `exp_results/analysis/per_layer/20260714_1542/per_layer_summary.txt`。

Llama 使用配置中的 `output_root=exp_results/llama2_7b_benchmark`，每组目录为：

```text
exp_results/llama2_7b_benchmark/<tag>/
  pipeline_manifest.json
  runkv_feedback/       # manifest、component JSONL、nsys、SQLite、prehook_timing
  tightllm_replay/      # 同上
exp_results/analysis/llama_per_layer/<tag>/
exp_results/logs/llama2_7b_benchmark_batch/<tag>/pipeline.log
```

已有的 Llama 1k 完整标签为 `20260729_145152_j001_llama2-7b-context1k-bs32-decode64_Llama-2-7b-hf-8k_feedback_pb32_bs32_p1000_d64`，其 `pipeline_manifest.json` 同时列出两系统产物。

history 中旧的 `configs/benchmark_batch.example.json` 后来改名为 `configs/benchmark_batch_opt_2.7b.json`。单独运行 `run_benchmark_pipeline.py` 时，当前默认是 OPT-6.7B、`B32/P2000/D32`、prefix `10000`、GPU utilization/staging fraction `0.9/0.6`；不能把这些默认值套到上述 batch 上。

### 3.3 分阶段 IO / SM 资源竞争

已完成范围：OPT 的 IO/SM，以及 Llama-2-7B 的 IO。两个系统使用相同 workload 和 resource schedule，每组通常重复 1 次。Llama SM 归入第 4 节。

全部当前配置的启动命令：

```bash
python "$ROOT/scripts/run_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_opt_2.7b_io.json"
python "$ROOT/scripts/run_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_opt_2.7b_sm.json"
python "$ROOT/scripts/run_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_opt_6.7b_io.json"
python "$ROOT/scripts/run_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_opt_6.7b_sm.json"
python "$ROOT/scripts/run_llama_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_llama2_7b_io.json"
```

| 参数 | OPT-2.7B IO/SM、OPT-6.7B IO | OPT-6.7B SM | Llama IO |
| --- | --- | --- | --- |
| B/P/D 四组 | 32/1000/128；32/2000/64；32/4000/32；16/8000/32 | 相同 | 8/1000/128；4/2000/64；2/4000/32；1/8000/32 |
| CPU GiB / fraction | 47.9 / 0.8 | 67.9 / 0.8 | 8 / 0.8 |
| GPU utilization / staging fraction | 0.7 / 0.9 | 0.7 / 0.9 | 0.9 / 0.9 |
| buffers / prefix blocks | 3 / 10000000 | 3 / 10000000 | 3 / 32 |
| max_num_seqs / max_staging_blocks | 自动 | 自动 | 8 / 640 |

IO 参数：`kind=io`、`clock=step`、`pattern=0:24,16:12,32:8`、`mode=process`、`direction=h2d`、buffer `256 MB`、`io_max_gbps=24`。

SM 参数：`kind=sm`、`clock=step`、`pattern=0:0,16:50,32:33.3`、`mode=process`、`matrix_size=4096`、`dtype=float16`。

在 process 模式下，pattern 的正数表示期望的 benchmark 可用资源份额：

- IO 基准 24 GB/s 时，24/12/8 对应额外 0/1/2 个持续传输的竞争进程。
- SM 100/50/33.3 对应额外 0/1/2 个竞争进程。
- `0` 在 process 模式下也表示不启动竞争进程，不表示可用带宽或算力为零。
- 这是通过竞争进程数量模拟的份额，不保证实际带宽精确等于目标值。

当前 OPT-2.7B IO 显式配置 allocation 为 RunKV `concentrate`、TightLLM `spread`；其余 OPT staged 配置没有覆盖时使用脚本默认 RunKV `spread`、TightLLM `concentrate`。Llama IO 为 `concentrate/spread`。不能统一套用普通 benchmark 的策略。

结果路径：

| 产物 | OPT | Llama |
| --- | --- | --- |
| 原始运行 | `exp_results/staged_offline_pilot/<pattern>/<tag>/{runkv_feedback,tightllm_replay}/r0/` | `exp_results/staged_llama2_7b/<pattern>/<tag>/{runkv_feedback,tightllm_replay}/r0/` |
| pipeline manifest | `exp_results/manifests/staged_resource/<pattern>_<tag>.json` | `exp_results/manifests/staged_resource_llama2_7b/<pattern>_<tag>.json` |
| 分阶段分析 | `exp_results/analysis/staged_resource/<pattern>/<tag>/` | `exp_results/analysis/staged_resource_llama2_7b/<pattern>/<tag>/` |
| 逐层分析 | `exp_results/analysis/staged_resource_per_layer/<pattern>/<tag>/r0/` | `exp_results/analysis/staged_resource_per_layer_llama2_7b/<pattern>/<tag>/r0/` |
| batch 日志 | `exp_results/logs/staged_resource_benchmark_batch/<tag>/staged_pipeline.log` | `exp_results/logs/staged_resource_benchmark_llama2_7b/io/<tag>/staged_pipeline.log` |

每个 `r0/` 通常包含 `manifest.json`、`run.log`、`pressure.csv`、`resource_steps.jsonl`、component JSONL、Nsight/SQLite 和 prehook timing。

已有完成批次可用以下 pattern/tag 定位：

- OPT-2.7B IO：`io_pcie12_step` / `20260729_110917_j001_opt2p7b-bs32-p1k-d128_opt-2p7b-8k_io_pcie12_step_bs32_p1000_d128`。
- OPT-6.7B SM：`sm_process50_step` / `20260806_155146_j004_opt6p7b-bs16-p8k-d32_opt-6p7b-8k_sm_process50_step_bs16_p8000_d32`。
- Llama IO：`llama2_7b_io_pcie_step` / `20260729_160954_j001_llama2-7b-bs8-p1k-d128_Llama-2-7b-hf-8k_llama2_7b_io_pcie_step_bs8_p1000_d128`。

7 月 14—15 日还手动试过 `throttle/process`、`0:0,20:8`、不同请求数和上下文长度，涉及 OPT-1.3B。这些探索的参数应逐次查看原始 manifest。单独启动 staged 脚本的当前默认 workload 为 OPT-1.3B、`B64/P2000/D256`，与上面的 JSON 不同。

### 3.4 同卡双推理任务 case study

用途：比较 RunKV+RunKV 与 TightLLM+TightLLM 在同一 GPU 上的吞吐、重叠区间和公平性。任务 A 达到 step 20 后启动任务 B。

history 中记录的是无参数入口；下面显式列出当前关键默认值：

```bash
python "$ROOT/scripts/run_concurrent_case_study.py" \
  --model /data/models/opt-2.7b-8k \
  --tightllm-profile-path "$ROOT/exp_results/tightllm_profiles/ubuntu/opt-2.7b-8k.json" \
  --num-prompts 32 --prompt-words 2000 --max-tokens 64 \
  --prefix-blocks 10000 --trigger-step 20 \
  --gpu-memory-utilization 0.25 --gpu-memory-fraction 0.9 \
  --num-device-buffers 3 --max-num-seqs 4 --max-staging-blocks 1152 \
  --cpu-memory-gb 25 --cpu-memory-fraction 0.8
```

默认启用 nsys，sample 为 `none`，并导出 SQLite。参数中的内存预算按每个推理任务配置。

结果根目录：`exp_results/concurrent_case_study/<tag>/`，包含 `case_summary.json` 和 `runkv_runkv/`、`tightllm_tightllm/` 两组子目录。

已完成证据：

```text
exp_results/concurrent_case_study/20260716_101457_opt-2.7b-8k_step20_bs32_p2000_d64/case_summary.json
```

该文件记录 `status=complete`，四个推理进程 return code 均为 0，且确认了实际 overlap。更早的其他 case-study 目录不能自动视为成功。

### 3.5 RunKV / TightLLM 与 Flux 共跑

有两种入口，产物不同。

**分别对两个系统施加 Flux 竞争**：history 记录过 OPT batch-with-flux 启动。当前使用已重命名的配置：

```bash
python "$ROOT/scripts/run_benchmark_batch_with_flux.py" \
  "$ROOT/configs/benchmark_batch_opt_2.7b.json"
```

每个 RunKV/TightLLM 子测试前启动 Flux，结束后停止；测试负载来自普通 batch JSON，输出沿用第 3.2 节的 OPT 路径。这条 batch history 本身不证明所有 case 均完成。

**把 RunKV 与 Flux 采到同一份 Nsight 报告**：7 月 23 日有 OPT-2.7B 成功产物。

```bash
python "$ROOT/scripts/run_runkv_flux_combined_nsys.py" \
  "$ROOT/configs/benchmark_batch_opt_2.7b.json" \
  --test-index 1 --run-tag "$RUN_TAG"
```

`--test-index` 从 1 开始；此入口只跑所选 RunKV case，默认 `--nsys-sample cpu`，关闭 RunKV 内层 nsys，由外层采集整个进程树。历史第一组实际为 `B32/P1000/D128`、prefix `10000000`、GPU utilization/staging `0.6/0.9`、CPU `47.9 GiB`；当前 JSON 已有变化。

Flux 管理器位于仓库外 `/home/lyc/inference/val`，脚本会加载 `diffusion/.venv/bin/activate`，调用 `diffusion/manage_flux_contender.py start --gpu 0`，结束时调用 `stop`。

结果为：

```text
exp_results/runkv_flux_nsys/<tag>/
  combined_manifest.json
  runkv_manifest.json
  runkv_flux_combined.nsys-rep
  nsys.log
  runkv/
```

已完成标签是 `20260723_151253_j001_opt2p7b-bs32-p1k-d64_opt-2p7b-8k_cpu47p9_gu0p6_gf0p9_rkcopy-segment_rkalloc-concentrate_tlcopy-segment_tlalloc-spread_bs32_p1000_d128`，其 `combined_manifest.json` 记录 `profile_return_code=0`。

### 3.6 组件消融：稳定环境与 IO 竞争

已完成：2026-08-10 的 OPT-2.7B，两个环境各 4 负载 × 5 变体。最终报告均为 20 行 `Valid=yes`。

| variant | ASYNC_PLAN_BUILD | H2D_COPY_MODE | USE_STATE_MACHINE |
| --- | --- | --- | --- |
| full | 1 | segment | 1 |
| no_async_plan | 0 | segment | 1 |
| no_segment_dma | 1 | gather | 1 |
| no_segment_dma_uva | 1 | uva | 1 |
| no_state_machine | 1 | segment | 0 |

启动与分析示例：

```bash
python "$ROOT/scripts/run_no_throttle_component_ablation.py" \
  --settings-config "$ROOT/configs/benchmark_batch_opt_2.7b.json" \
  --run-tag "$RUN_TAG"

python "$ROOT/tools/analyze_component_ablation.py" \
  --run-tag "$RUN_TAG" --final-only \
  --output-dir "$ROOT/exp_results/analysis/component_ablation/no_throttle/$RUN_TAG"

python "$ROOT/scripts/run_io_contention_component_ablation.py" \
  --settings-config "$ROOT/configs/staged_resource_benchmark_opt_2.7b_io.json" \
  --disable-nsys --run-tag "$RUN_TAG"
```

history 中稳定环境的最终原命令标签为 `--run-tag opt27b_ablation_final`。IO 最终批次为 `io_contention_ablation_opt27b_20260810_nsysfree`，没有 nsys；上面的 IO 命令是依据当前入口和该批 manifest 重建的方式。

| 历史最终批次 | B/P/D 四组 | 公共参数与采集方式 |
| --- | --- | --- |
| no_throttle | 32/1000/64；32/2000/64；32/4000/32；16/8000/32 | prefix=10000000、CPU=47.9 GiB、GPU utilization/staging=0.7/0.9、allocation=concentrate；有 nsys |
| io_contention | 32/1000/128；32/2000/64；32/4000/32；16/8000/32 | 同上内存和策略，叠加第 3.3 节 IO schedule；最终批次无 nsys |

可用 `--workloads 1k 2k`、`--variants full no_async_plan` 筛选；`--smoke` 改为 1 请求、64 words、2 输出 tokens；`--resume` 在同一 run tag 下跳过完整产物。

结果与完成证据：

```text
exp_results/ablation/no_throttle/<workload>/<variant>/opt27b_ablation_final/
exp_results/analysis/component_ablation/no_throttle/final_ablation_results.md

exp_results/ablation/io_contention/<workload>/<variant>/io_contention_ablation_opt27b_20260810_nsysfree/
exp_results/analysis/component_ablation/io_contention/io_contention_ablation_opt27b_20260810_nsysfree/final_ablation_results.md
```

历史最终批次的原始目录包含 `manifest.json`、`run.log`、component JSONL、prehook timing；IO nsysfree 批次另有 `step_metrics.jsonl`、`pressure.csv`、`resource_steps.jsonl`，开启 nsys 时另有报告和 SQLite。历史 no_throttle 最终批次没有 `step_metrics.jsonl`；当前消融入口的新运行会生成该文件。新运行的最后一级目录为自己的 `RUN_TAG`。

稳定环境最终报告使用 Nsight inference-loop 吞吐；IO 无 nsys 批次使用 `step_metrics_span`，没有 kernel timeline，不能把它的 pipeline bubble 缺失误读为零。

### 3.7 Spec ON/OFF：异步计划构建验证

这里的 Spec 是后台 speculative replay-plan build，对应 `layer_recompute_async_plan_build`，不是 token speculative decoding。

已完成范围：OPT-2.7B、OPT-6.7B、Llama-2-7B；稳定与 IO 竞争两个环境；1k/2k/4k/8k；ON/OFF 配对。OPT-6.7B、Llama 在 8 月 31 日补测，OPT-2.7B 使用已有严格匹配记录。报告选取每模型、环境、负载各一对，共 24 对（48 次独立运行）。

history 没有保留这一组的原始命令。下面依当前消融入口和 manifest 重建，显式设置输出根及关闭 nsys：

```bash
python "$ROOT/scripts/run_no_throttle_component_ablation.py" \
  --settings-config "$ROOT/configs/spec_claim_opt_6.7b_stable.json" \
  --variants full no_async_plan --disable-nsys \
  --output-root "$ROOT/exp_results/ablation/spec_claim/opt_6.7b/stable" \
  --run-tag "$RUN_TAG"

python "$ROOT/scripts/run_io_contention_component_ablation.py" \
  --settings-config "$ROOT/configs/spec_claim_opt_6.7b_io.json" \
  --variants full no_async_plan --disable-nsys \
  --output-root "$ROOT/exp_results/ablation/spec_claim/opt_6.7b/io_contention" \
  --run-tag "$RUN_TAG"
```

Llama 分别替换为 `spec_claim_llama2_7b_stable.json`、`spec_claim_llama2_7b_io.json`，输出根中的 `opt_6.7b` 替换为 `llama2_7b`；入口会自动选择 Llama runner。

| 配置 | B/P/D 四组 | 内存、prefix 与策略 |
| --- | --- | --- |
| OPT-6.7B stable / IO | 32/1000/128；32/2000/64；32/4000/32；16/8000/32 | CPU47.9、GPU0.7/0.9、prefix10000000、allocation **spread** |
| Llama stable | 32/1000/64；32/2000/32；16/4000/32；8/8000/32 | CPU47.9、GPU0.7/0.9、prefix32/64/128/256、allocation concentrate |
| Llama IO | 8/1000/128；4/2000/64；2/4000/32；1/8000/32 | CPU8、GPU0.9/0.9、prefix32、max_num_seqs8、staging640、allocation concentrate |

IO 使用 `0:24,16:12,32:8` 的 process H2D 竞争。ON=`full`，OFF=`no_async_plan`，H2D 都是 segment、状态机都开，其余配置应严格匹配。关闭异步构建仍保留 steady-successor 快速路径。

原始补测结果：

```text
exp_results/ablation/spec_claim/{opt_6.7b,llama2_7b}/{stable,io_contention}/<workload>/{full,no_async_plan}/spec_claim_4090_20260831/
```

该目录模板使用 shell 花括号表示候选项，不是字面目录名。每组有 manifest、完成日志、component/step/prehook 数据。Llama IO 1k full 的 `run.log` 明确记录 `LLAMA replay run finished.` 和 131 个 prehook steps。

分析结果位于：

- [配对范围与曲线报告](../analysis_artifacts/spec_on_off_slope_curves_4090_all_models/analysis_report.md)。
- [三模型 claim 指标](../analysis_artifacts/spec_claim_metrics_4090_all_models/report.md)。
- [TRACKING 收敛分析](../analysis_artifacts/spec_tracking_performance/tracking_report.md)。
- [相邻 gain 稳定性分析](../analysis_artifacts/spec_slope_stability/slope_stability_report.md)。

这些报告中的控制器 observation 按模型层计数，不等于 inference-loop decode step；比较吞吐和收敛指标时应保持统计口径一致。

### 3.8 真实分布 segment-DMA 微基准

分为真实推理 capture 和独立传输重放。8 月 27 日 history 记录 capture 后用 `--last-records 64 --warmup 5 --iterations 50` 重放；8 月 30 日留存的三模型正式结果使用另一套窗口与 30 次迭代。

可复用的 capture 命令，显式选择 32 请求 × 1000 words × 64 输出 tokens，并覆盖 observation runner 的旧默认输出路径；这不是 8 月 27 日 history 的逐字命令：

```bash
RUNKV_CAPTURE_BLOCK_DIST=1 \
RUNKV_CAPTURE_BLOCK_DIST_DIR=/tmp/runkv-block-dist \
RUNKV_CAPTURE_BLOCK_DIST_MAX=500 \
DRY_RUN=0 H2D_COPY_MODE=segment \
MODEL=/data/models/opt-2.7b-8k \
NUM_PROMPTS=32 PROMPT_WORDS=1000 MAX_TOKENS=64 \
OUTPUT_DIR="$ROOT/exp_results/dma_capture/$RUN_TAG" \
python "$ROOT/examples/offline_inference/run_opt_feedback_observation.py"

python "$ROOT/benchmarks/kernels/benchmark_runkv_segment_dma.py" \
  --capture /tmp/runkv-block-dist \
  --last-records 64 --warmup 5 --iterations 50 \
  --output-dir /tmp/runkv-segment-dma
```

当前默认 methods 只有 `real packed contiguous`，并非自动包含 single DMA：

| method | 保持或改变的条件 |
| --- | --- |
| real | 保留实际源/目标地址空洞和段数，每段分别提交 K/V copy |
| packed | 消除地址空洞，保持 payload 和分段提交方式 |
| contiguous | 连续布局，用于对照分段提交开销 |
| single_dma | 把相同有效 K+V 字节展平为一次 copy；不包含生产环境打包成本 |
| per_block | 按 block 提交，额外可选对照 |

8 月 30 日测量：三个模型的 capture 都是 `P1000/B32/D64`，取中间层 L16 的 records 16—78，共 63 个 decode DMA 记录，包含 drain；5 warmups、30 timed iterations。以下是按 summary 重建的 OPT-6.7B 单 DMA 对照命令：

```bash
python "$ROOT/benchmarks/kernels/benchmark_runkv_segment_dma.py" \
  --capture "$ROOT/analysis_artifacts/segment_dma/opt6p7b_p1k_bs32_d64/capture/block_dist_1737752.json" \
  --last-records 63 \
  --methods real single_dma --warmup 5 --iterations 30 \
  --output-dir "/tmp/runkv-segment-dma-$RUN_TAG"
```

分解对照改为 `--methods real packed contiguous`。8 月 30 日原始采样顺序、硬件与数据口径见 [DMA 测量报告](../analysis_artifacts/segment_dma/README.md)。

| 模型 | 已有 capture 路径（相对 `analysis_artifacts/segment_dma/`） |
| --- | --- |
| OPT-2.7B | `opt2p7b_p1k_bs32_d64/capture/block_dist_1738481.json` |
| OPT-6.7B | `opt6p7b_p1k_bs32_d64/capture/block_dist_1737752.json` |
| Llama-2-7B | `llama2_7b_p1k_bs32_d64/capture/block_dist_1738067.json` |

各模型目录下的 `microbenchmark/summary.json`、`single_dma_microbenchmark/summary.json` 和 CSV 是完成产物；汇总表为 `analysis_artifacts/segment_dma/capture_microbenchmark_summary.csv`。新微基准的结果目录由 `--output-dir` 指定。

## 4. 仅确认脚本入口、未找到完成证据的测试

本节命令用于说明如何启动，不代表这些测试已执行或通过。已有 pytest cache 只能辅助定位用例，不能替代完整通过记录。

### 4.1 Llama SM 与 Flux

```bash
python "$ROOT/scripts/run_llama_staged_resource_benchmark_batch.py" \
  "$ROOT/configs/staged_resource_benchmark_llama2_7b_sm.json"

python "$ROOT/scripts/run_llama_benchmark_batch_with_flux.py" \
  "$ROOT/configs/benchmark_batch_llama2_7b.json"
```

Llama SM 的负载、内存与第 3.3 节 Llama IO 相同，改为 `kind=sm`、`pattern=0:100,16:50,32:33.3`、`mode=process`、matrix size 4096、float16；allocation 为 RunKV concentrate / TightLLM spread。预期输出为 `exp_results/staged_llama2_7b/llama2_7b_sm_process_step/<tag>/`，日志在 `exp_results/logs/staged_resource_benchmark_llama2_7b/sm/`。

Llama Flux 沿用普通 Llama JSON 参数和第 3.2 节输出结构，并依赖第 3.5 节仓库外 Flux 服务。未找到两类的本地完成产物。

### 4.2 旧 RunKV / vanilla vLLM 综合套件

入口：[benchmarks/runkv_benchmark.sh](../benchmarks/runkv_benchmark.sh)。

```bash
MODEL=/data/models/opt-2.7b-8k CUDA_DEVICE=0 \
RESULTS_DIR="$ROOT/exp_results/vanilla_vs_runkv/$RUN_TAG" \
bash "$ROOT/benchmarks/runkv_benchmark.sh" all
```

当前默认：Qwen2.5-1.5B-Instruct、GPU 2、1000 prompts、seed 42、GPU utilization 0.5、max_num_seqs 30；RunKV 为 3 buffers、GPU staging fraction 1.0、CPU 30 GiB / fraction 0.3。

| 子测试 | 默认负载网格 |
| --- | --- |
| throughput | input 128/512/1024/2048 × output 128/256/512 |
| latency | batch 1/4/8/16/32 × input 128/512/1024；output 128、warmup 5、iterations 30 |
| serving | QPS 1/2/4/8/16；input 512/output 128，200 prompts，port 8000 |
| prefix | prefix 256/512/1024 × repeat 5/10/20；50 prompts，input 512:1024/output 128 |
| sharegpt | 500 prompts |
| 长序列 | total 4096/8192/16384/32768 × output 128/256/512；100 prompts，跳过超模型长度的组合 |
| 高并发 | max_num_seqs 16/32/64/128/256；input 512/output 128，500 prompts |

支持 `all`、`resume`、`throughput`、`latency`、`serving`、`prefix`、`sharegpt`、`download_sharegpt`；长序列和高并发没有独立子命令，由 all/resume 调用。

默认结果目录是 `benchmarks/benchmark_results/`；可通过 `RESULTS_DIR` 覆盖。产物包括各类 JSON、日志和失败标记。尚未找到本地完成结果。

### 4.3 PCIe、NVMe、限速和旧 DMA 微基准

以下给出当前关键默认参数与输出位置：

| 入口 | 参数和覆盖范围 | 输出 |
| --- | --- | --- |
| [benchmark_gpu_cpu_bandwidth.py](../scripts/benchmark_gpu_cpu_bandwidth.py) | cuda:0；H2D/D2H/双向 × pinned/pageable；1/4/16/64/128/256/512/1024 MiB，1 stream，每项 3 秒、warmup 5 | 见下方显式 `--output-dir`，结果含 JSON/Markdown |
| [benchmark_dual_gpu_pcie_contention.py](../scripts/benchmark_dual_gpu_pcie_contention.py) | cuda:0/1；solo/concurrent；256 MiB、5 秒、warmup 10；可选 `--per-iter-sync` | 见下方显式 `--output-dir`，结果含 JSON/Markdown |
| [benchmark_gpu_nvme_contention.py](../scripts/benchmark_gpu_nvme_contention.py) | cuda:0、GPU buffer 256 MiB、8 秒；fio 文件 8G、bs 1M、iodepth 32、numjobs 1、libaio | `exp_results/pcie_nvme_contention/<timestamp>/{results.json,summary.md}` |
| [benchmark_throttled_io_bandwidth.py](../scripts/benchmark_throttled_io_bandwidth.py) | cuda:0、64 MiB、32 copies；target 2/5/10/20 GB/s；验证 IO 限速和 overlap | 默认 stdout |
| [bench_dma_segments.py](../examples/offline_inference/bench_dma_segments.py) | 旧 CPU gather+DMA / segment 对照；synthetic 默认 256 blocks、skip 0.3、float16、warmup 10、iters 50 | 默认 stdout；不是第 3.8 节真实 capture 微基准的同一入口 |

```bash
python "$ROOT/scripts/benchmark_gpu_cpu_bandwidth.py" \
  --output-dir "$ROOT/exp_results/gpu_cpu_bandwidth"

# 双 GPU 测试需要暴露两张卡，覆盖公共环境中的单卡设置。
CUDA_VISIBLE_DEVICES=0,1 \
python "$ROOT/scripts/benchmark_dual_gpu_pcie_contention.py" \
  --output-dir "$ROOT/exp_results/dual_gpu_pcie_contention"

python "$ROOT/scripts/benchmark_gpu_nvme_contention.py" \
  --file /mnt/nvme0n1p1/runkv_pcie_contention/testfile.bin \
  --output-dir "$ROOT/exp_results/pcie_nvme_contention"

python "$ROOT/scripts/benchmark_throttled_io_bandwidth.py"

python "$ROOT/examples/offline_inference/bench_dma_segments.py" \
  --synthetic --num-blocks 256 --skip-ratio 0.3 \
  --cpu-shape 512,2,16,8,128
```

前两个 bandwidth 入口会在输出根下新建时间戳目录。NVMe 测试的 `--file` 是实际 fio 测试文件路径，脚本会创建/写入，正常结束默认删除；不要指定需要保留的数据文件。

### 4.4 planner dry-run 上下文失衡诊断

```bash
bash "$ROOT/scripts/run_dryrun_context_imbalance.sh"
```

默认 OPT-2.7B、prefix 10000、3 buffers、状态机开启、无资源压力；`DRY_RUN=1` 表示 planner 只观察，但仍执行真实 GPU 推理。

| prompt words | 请求数 / 输出 tokens | GPU utilization / staging fraction | CPU GiB / fraction |
| --- | --- | --- | --- |
| 1000、2000 | 32 / 128 | 0.7 / 0.95 | 46.566 / 0.3 |
| 4000 | 32 / 32 | 0.8 / 0.7 | 93.132 / 0.6 |
| 8000 | 16 / 32 | 0.8 / 0.6 | 93.132 / 0.6 |

默认输出 `exp_results/dryrun_context/<RUN_GROUP>/<1k|2k|4k|8k>/`，分析为 `exp_results/analysis/dryrun_imbalance/<RUN_GROUP>/`。可用环境变量 `RUN_GROUP`、`OUTPUT_ROOT`、`ANALYSIS_OUTPUT_DIR`、`CONTEXTS` 覆盖；默认自动调用 `tools/analyze_dryrun_imbalance.py --only-modal-step-shape`。

### 4.5 单请求、并发与动态回放 E2E

| 入口 | 当前主要默认参数 / 验证点 |
| --- | --- |
| [test_runkv_e2e_single_req.py](../tests/v1/kv_offload/test_runkv_e2e_single_req.py) | OPT-1.3B、256 output、max_num_seqs 8、GPU utilization/staging 0.5/0.85、CPU30/0.3；单请求 offload 与 baseline |
| [test_runkv_e2e_concurrent.py](../tests/v1/kv_offload/test_runkv_e2e_concurrent.py) | OPT-1.3B、100 requests、16—256 output、max_num_seqs256、GPU0.4/0.85、CPU30/0.3；并发、stress、baseline |
| [test_dynamic_replay_e2e_concurrent.py](../tests/v1/kv_offload/test_dynamic_replay_e2e_concurrent.py) | OPT-2.7B、100 requests、16—256 output、GPU0.8/0.9、CPU40/0.4；动态回放计划与 baseline |
| [test_feedback_planner_e2e_concurrent.py](../tests/v1/kv_offload/test_feedback_planner_e2e_concurrent.py) | OPT-2.7B、64 requests、16—32 output、prefix1000、GPU0.9/0.9、CPU40/0.4；feedback 全环与 observe-only |
| [test_layer_recompute_e2e.py](../tests/v1/kv_offload/test_layer_recompute_e2e.py) | pytest；单请求和并发下逐 token 对比 vanilla / RunKV baseline / recompute |

这些脚本的默认模型未必是扩展上下文后的 `-8k` 目录；复现时显式指定本地模型。前三类 CLI 与 feedback CLI 直接用 Python 启动，不应把所有文件都当作普通 pytest 用例运行。

单请求、并发 offload、feedback 全环的启动示例，负载使用对应脚本默认值：

```bash
python "$ROOT/tests/v1/kv_offload/test_runkv_e2e_single_req.py" \
  --model /data/models/opt-2.7b-8k --compare-baseline

python "$ROOT/tests/v1/kv_offload/test_runkv_e2e_concurrent.py" \
  --model /data/models/opt-2.7b-8k --compare-baseline

python "$ROOT/tests/v1/kv_offload/test_feedback_planner_e2e_concurrent.py" \
  --model /data/models/opt-2.7b-8k --compare-baseline
```

Llama dynamic replay 的小负载验证命令来自现有项目文档：

```bash
python "$ROOT/tests/v1/kv_offload/test_dynamic_replay_e2e_concurrent.py" \
  --model /data/models/Llama-2-7b-hf-8k \
  --enable-dynamic-replay --compare-baseline \
  --num-requests 2 --min-tokens 4 --max-tokens 4 \
  --max-num-seqs 2 --max-model-len 256 \
  --gpu-memory-utilization 0.9 --cpu-memory-gb 4 \
  --cpu-memory-fraction 0.1 --max-staging-blocks 64 \
  --layer-recompute-io-prefix-blocks 1

VLLM_RUNKV_E2E_MODEL=/data/models/opt-2.7b-8k \
python -m pytest "$ROOT/tests/v1/kv_offload/test_layer_recompute_e2e.py" -v -s
```

主要结果为 stdout 的请求完成情况、token IDs 和 baseline 差异；没有统一 benchmark manifest 输出根。需保存本次日志时，可自行重定向到新的 `exp_results/logs/` 文件。现有证据不足以断言这些 E2E 全部通过。

### 4.6 定向单元测试与分析工具回归

| 测试组 | 主要文件与覆盖 |
| --- | --- |
| KV offload / async | `tests/v1/kv_offload/test_runkv_offload.py`、`test_runkv_async.py`：block mapping、ring buffer、skip/load、gather/UVA、异步 load/flush |
| replay plan / metadata / forward | 同目录 `test_opt_dynamic_replay*.py`、`test_llama_dynamic_replay_forward.py`：对齐、slot mapping、CPU fill/GPU reuse、metadata 重用、Llama position/residual |
| controller / allocation | `test_imbalance_controller.py`、`test_replay_allocation_policy.py`：状态转换、deadband、gain、预算边界、spread/concentrate |
| profiler / TightLLM | `tests/v1/profiling/test_tightllm_offline_profiler.py`、`test_tightllm_ilp_planner.py`：架构公式、profile schema、planner |
| runner / benchmark | `tests/v1/kv_offload/test_llama_benchmark_runner.py`、`tests/benchmarks/test_llama_benchmark_scripts.py`、`test_resource_pressure_controller.py`、`test_runkv_segment_dma.py` |
| 分析工具 | `tests/tools/test_analyze_component_ablation.py`、`test_analyze_spec_slope_stability.py`、`test_analyze_spec_tracking_performance.py`、`test_plot_spec_on_off_slope_curves.py` |

示例：

```bash
python -m pytest \
  "$ROOT/tests/v1/kv_offload/test_imbalance_controller.py" \
  "$ROOT/tests/v1/kv_offload/test_replay_allocation_policy.py" \
  "$ROOT/tests/benchmarks/test_runkv_segment_dma.py" \
  "$ROOT/tests/tools/test_analyze_component_ablation.py" -q
```

默认输出到终端，pytest 可能生成 `.pytest_cache/`；如需独立测试报告，可增加 `--junitxml=/tmp/runkv-tests.xml`。pytest cache 的用例列表不代表这些用例在同一批次中全部通过。

## 5. 历史失败记录与复现时的区别

以下是找到的失败例子，不计入完成批次：

| 历史路径（相对 `exp_results/`） | 记录的问题 |
| --- | --- |
| `staged_offline_pilot/io_multistage_step/20260714_1717/runkv_feedback/r0/run.log` | 缺少 `benchmarks.runkv_resource_pressure` 模块 |
| `staged_offline_pilot/io_multistage_step/20260714_1712/runkv_feedback/r0/run.log` | KV 显存不足 |
| `staged_offline_pilot/io_multistage_step/20260714_2024/runkv_feedback/r0/run.log` | CUDA OOM |
| `ablation/no_throttle/4k/no_segment_dma/20260810_complete/run.log` | 退出码 137；后续 `opt27b_ablation_final` 有完成产物 |

复查历史时应保留以下区别：

1. 两个 OPT normal JSON 的 8k case 名称含 `bs32`，实际 `batch_size=16`；OPT-6.7B normal 第二项名称含 `d64`，实际 `decode_length=32`。
2. OPT-6.7B staged SM 第一项名字写 `p2k-d64`，实际为 `P1000/D128`，且与第二项重名；结果名与工作负载应以 manifest 字段核对。
3. `PREFIX_BLOCKS` 控制 IO prefix，不是 replay 数量；极大初始值不表示最终不会由 feedback 改变回放预算。
4. 外层 `--dry-run` 只展开命令；observation 的 `DRY_RUN=1` 是 planner observe-only，仍执行推理。
5. Llama 使用 `the` 构造 prompt，1000/2000/4000/8000 words 在本地 tokenizer 下约为 1013/2013/4013/8013 tokens。直接用 OPT 的 `replay` 会产生不同 token 长度。
6. `gpu_memory_utilization` 与 staging `gpu_memory_fraction` 是两个参数；动态回放的 CPU 总预算同时包含 KV 与 hidden-state backing store。
7. 直接调用 `run_opt_feedback_observation.py` 的默认 `OUTPUT_DIR` 仍指向旧仓库 `/home/lyc/inference/vllm/exp_results/opt_feedback_observation`；pipeline 会覆盖它。独立调用时应像第 3.8 节一样显式设置输出目录。
8. 有 `.nsys-rep` 不等于整个 paired pipeline 成功；应结合 return code、完成日志、manifest 和预期分析文件判断。无 nsys 批次也可以有有效的 step/组件分析结果。
