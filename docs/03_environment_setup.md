# 阶段 C：独立环境准备（PARTIAL，后台进行中）

2026-10-24。不能将本页当作安装、import或pipeline验收成功。

在阶段B硬件审计后，按用户“持续推进”授权启动独立setup。没有conda可执行文件，因此下载公开micromamba，在 `/data/local/envs/z0` 创建Python3.10环境。中性环境前缀是可见命令行隐私规则所需；实际已创建Python3.10.22。现有系统Python/环境/驱动/系统CUDA均不修改。

项目根 `/data/local/ViSkill`；源码在 `src/`，由本地审计工作树传输，无额外GitHub克隆。工具 `/data/local/tools/z0`，临时 `/data/local/tmp/z0`，都计入100GB项目预算。安装进程CUDA_VISIBLE_DEVICES为空，不使用GPU。后续仅物理卡3/4/5。

实际安装代码见 `scripts/prepare_environment_worker.py`：先torch2.8.0/torchvision0.23.0的官方cu128 wheel，再固定Transformers4.56.1及FrozenLake/veRL公共依赖，最后以`--no-deps -e`安装bundled veRL与vagen、运行pip check、导出freeze。安装不运行install.sh/install_render.sh，不安装PrimitiveSkill/AI2-THOR。独立FA包和SGLang尚需后续兼容性处理，不把Transformers可运行当PPO就绪。

固定下载模型revision：

- Base：`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`，目标`models/b0`。
- Release：`hongxingli/ViSkill-FrozenLake@edafb2b71f73afb1c35b82d264346c2ae2b5981e`，目标`models/r0`（包括skill library）。

下载仅公开资源，`token=False`，不需要真实API key，不进行模型API推理。关闭HF遥测、Ray usage stats、wandb；不自动上传。若依赖含新的外部访问行为，后续网络边界检查必须记录。

监督进程PID2545403，worker PID2545406（仅历史启动标识，后续依据receipt与当前状态判断）。完整命令行已检查为中性python3 -u -c入口，pip子进程为中性环境路径；项目路径通过环境传递。没有GPU实验进程。

记录/判定：

| 记录 | 路径 |
|---|---|
| 原始开始时间、PID、预算和路径 | `/data/local/ViSkill/state/setup_receipt.json` |
| 每步安装命令及退出码 | `/data/local/ViSkill/state/setup_commands.json` |
| 安装/下载stdout与错误 | `/data/local/ViSkill/logs/setup.log` |
| 实际分配存储与峰值 | `/data/local/ViSkill/state/setup_resource.json` |
| 监督进程最终退出 | `/data/local/ViSkill/state/SETUP_EXIT.json` |
| worker失败/完成 | `SETUP_FAILURE.json` / `SETUP_COMPLETE.json`（同state目录） |

独立supervisor每5秒检查本项目四处路径的实际分配字节，85GB停止写入并终止本次worker进程组，留15GB余量；setup硬超时6小时，后续修复不重置开始时间。100GB预算不是允许写满磁盘。无文件存在/大小检查替代未完成的功能验证。

安装结束后仍必须检查退出状态、pip check、冻结版本，执行全部指定imports和CUDA/BF16/NCCL/FA检查，再D真实图像VLM交互和E组件测试。既有v0环境的多卡NCCL超时记录必须保留，新环境未通过就不启动多卡PPO。阶段D/E未完成，不启动任何PPO。

## 2026-10-24 网络修复与阶段C部分验收

依赖安装与pip check已退出0，实际freeze和核心导入记录见`evidence/environment/`。首次模型下载因网络不可达失败，保留原日志和EXIT。`scripts/resume_setup_supervisor.py`只续传公开固定revision模型，通过远程loopback28791转发本机代理7897；共享原6小时截止/85GB停止线，绝不重建环境或重置时钟。修复单独receipt/EXIT/log与原失败并存。GPU4基础BF16/SDPA、NCCL world1与GPU3/4/5三rank all_reduce通过；FrozenLake视觉reset通过。SGLang/FA尚未安装，真实模型尚未加载，阶段C仍PARTIAL。

### PPO runtime attempt1: Blackwell extra Python mismatch

`sglang[blackwell]==0.5.2` failed resolution (EXIT1): its
`nvidia-cutlass-dsl==4.1.0` extra has only cp312 wheels, while this independent
project environment is Python3.10. Public package metadata verified at
https://pypi.org/pypi/nvidia-cutlass-dsl/4.1.0/json and
https://pypi.org/pypi/sglang/0.5.2/json. Failure artifacts are retained under
`docs/evidence/environment/runtime_attempt1/`; recorded peak40,678,797,312 bytes.

Repair uses the same pinned `sglang[srt]==0.5.2`, whose dependency list is identical
to the Blackwell extra except for that DSL package. This is a dependency-set change,
not proof of SM120 compatibility; actual FA and rollout runtime probes are pending.
The current BF16 Qwen route does not request FP4 kernels. All other core constraints
are preserved. Receipt `state/ppo_runtime_srt_receipt.json`, exit
`state/PPO_RUNTIME_SRT_EXIT.json`, log `logs/ppo_runtime_srt.log`; original setup
start1792772381.9046373/deadline1792793981.9046373 and85GB guard unchanged.

### Actual runtime checks and transitive dependency repair

SGLang[srt] installation and pip check completed EXIT0. FlashAttention2.8.3 on
physical GPU4 executed BF16 q/k/v attention plus backward: finite output/gradients,
gradient norms0.004283/0.004305/0.004486. This is a small kernel check, not actor or
critic PPO backward. All engine/FlashInfer imports passed, but true VLM engine
startup exposed CUTLASS4.8.1 API incompatibility (`cute.core.ThrMma`) in FA's cute
module. The newly resolved cuDNN frontend1.31 had pulled that newer DSL transitively.

Pinned cuDNN frontend1.13.0 and CUTLASS DSL4.2.0 (cp310 available), resolving
cuda-python/bindings to12.9.10; pip check passed. See runtime_pins evidence and
https://docs.nvidia.com/cutlass/4.2.1/overview.html for Python3.10 support introduced
in4.2. Model/core packages remain torch2.8.0, Transformers4.56.1, SGLang0.5.2.

Engine probe attempts are separately preserved: v1 missing processor registration
caused by that import conflict; v2 loaded model but probe loop was lost when Engine
installed uvloop policy; v3 fixed loop and reached generation but failed because
`ninja` was absent from the launcher PATH. v4 adds env/bin and system CUDA bin to
PATH, without changing installed CUDA. All attempts share original setup clock and
have max600s subprocess guards. Text attention=Triton, vision attention=SDPA are
explicit runtime deviations from the trainer's FA3 default, which is not assumed
to support this GPU. C cannot be called complete until actual generation passes.

The first FlashInfer import created only an8KiB cache directory tree and a zero-byte
log under the account cache. It is retained and counted conservatively as8KiB.
Subsequent compilation caches are explicitly under `/data/local/tools/z0/cache` so
compiler subprocess arguments use neutral paths; no large cache is permitted in
home. Existing earlier project cache directories remain within the budget roots.

v4实际生成PASS/EXIT0：SGLang加载base BF16后处理真实FrozenLake seed1图像，
生成44tokens、解析Right并由environment.step执行；prompt362tokens，
含首次JIT的生成延迟34.51s，包含加载总49.17s。不是完整episode或PPO。
权重占约7.33GiB，KV cache两部分各4.27GiB；单请求仍按mem_fraction_static=.25
预留缓存。模型/critic的PPO显存及FSDP切换仍待后续门槛通过后验证。

vagen.ray_trainer_skill和bundled async_sglang_server实际导入PASS。C所需基础
环境运行检查通过；保留退出时torch._inductor查找cutlass.CACHE_FILE的非致命
atexit警告（本次EXIT0），不把它当作训练成功。记录见runtime_engine_v4_path。


### Official agent-loop import gap repaired

The 2026-10-24 directional integration attempt exited before model loading because the bundled official agent-loop imports cachetools, which was absent from the manually pinned environment. Installed cachetools==5.5.2 with --no-deps into the isolated z0 prefix and retained the failed EXIT/log. The worker installation list now includes this pin; pip check reports no broken requirements and environment/pip_freeze.txt was refreshed. The repair shares the first integration deadline1792819854.7692268; no rollout was rerun from the failed import attempt. Actual loop execution is tracked separately and import success is not an E/PPO result.
