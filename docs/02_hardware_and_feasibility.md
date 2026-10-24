# 阶段 B：pro5000 实际硬件与可行性

时间：2026-10-24 Asia/Shanghai。原始采集时间见 `evidence/hardware/hardware_snapshot.json`（服务器UTC时间应加8小时）。用户最新指令覆盖先前资源选择：**仅使用物理GPU 3、4、5，最多3张；本项目新增实际存储不超过100,000,000,000字节**。此前50GB预算与0/1/2候选均作废，历史检查保留。独立实验24小时上限不变；12小时不在不意味着可以跳过科学门槛或启动300步训练。

## 1. 实测硬件与当前可用资源

| 项目 | 实际结果 | 限制 |
|---|---|---|
| GPU | 8×NVIDIA RTX PRO 5000 72GB Blackwell | 本任务只能用3/4/5，不得使用其余5张 |
| 单卡显存 | 73,415 MiB ≈71.69 GiB | free小于total-used，包含驱动保留，不假定全部可分配 |
| 用户指定卡 | 最近轻量检查GPU3 free72,187MiB；GPU4/5 free72,830MiB | GPU3有他人/其他项目既有小任务；不终止或修改，不要求独占；启动实际任务前再检查 |
| driver | 595.104.02；nvidia-smi显示CUDA13.2 | driver支持上限不是PyTorch运行时版本 |
| 系统CUDA toolkit | `/usr/local/cuda/bin/nvcc`，13.2.51 | `nvcc`不在默认PATH，首次命令退出127；不是系统未安装CUDA |
| CPU | 2×Xeon Gold6530，64物理核/128线程 | NUMA两节点 |
| RAM | 总754GiB，可用约709GiB | `free`仅4GiB不表示内存不足，大量可回收page cache |
| swap | 总8GiB，已用约467MiB | 不用swap代替GPU容量规划 |
| 项目存储 | `/data/local`位于`/data`，ext4，rw；可用约14TiB | 必须使用/data/local，不回退home/root；100GB是项目预算，不是磁盘剩余量 |
| /dev/shm | 约378GiB，基本空闲 | 容量足够不等于NCCL已经可用 |
| GPU topology | 4/5同NUMA1，3属NUMA0；3↔4/5为SYS，4↔5为NODE | 无NVLink记录；三卡不会自动比两卡更好 |
| 默认Python | 3.12.3，系统python | 不用于新项目安装；不得全局pip或覆盖已有环境 |
| 既有torch探针 | 只读借用`/data/local/envs/v0`：torch2.7.1+cu128、CUDA12.8、NCCL2.26.2 | 不是本项目独立环境，不修改它 |
| 既有另一环境 | s1有torch2.8.0、vLLM0.10.2 | 仅metadata查阅，不作为已验证的本项目环境 |

`/data/local`小型创建→写入→fsync→读取探针PASS，已仅删除本次临时文件；无历史数据清理。服务器连接、查询与探针均无训练。

## 2. 能力实测与负结果

证据：`gpu_capability_probe.json`、`gpu_capability_stderr.txt`。

- GPU compute capability为12.0；torch编译arch列表包含sm_120。
- BF16支持查询、64×64 BF16矩阵乘法、SDPA，以及强制PyTorch FLASH_ATTENTION backend的小张量探针均PASS，输出finite。该历史探针发生在用户改为345之前的GPU0；不再继续使用GPU0。
- 独立`flash_attn`包未安装。PyTorch Flash SDPA成功**不能**替代官方PPO loader所要求的`flash_attention_2`包检查。
- NCCL availability查询为true，但实际多卡collective未通过，不能据此宣称FSDP可以运行。
- 首次0/1双卡探针75秒超时，原始脚本使用fork，失败时本地异常使部分stdout未落盘；已确认仅本次三个`python -`PID及父子关系，清理了这些进程和本次rendezvous文件，无其他进程受到操作。
- 改用spawn后，345默认配置探针50秒超时；再使用中性文件入口、`NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1,NCCL_SOCKET_IFNAME=lo`进行定位，三个rank均进入分布式初始化后仍未返回all_reduce，同样50秒超时。日志分别见`nccl_three_gpu_probe.json`与`nccl_three_gpu_fallback.json`，两次均由独立远程supervisor终止自己的进程组。
- 以上不是OOM、不是PPO失败，也不能推断硬件故障或最终定位为网络/P2P问题。下一次不同软件栈测试必须保留这些负结果。未进行性能benchmark。

## 3. 模型、显存和checkpoint计算

估算依据是本次下载的官方config及safetensors index，未下载大权重时已能核对文件体积。固定base revision `66285546d2b821cf421d4f5eb2576359d3770cd3`；release revision `edafb2b71f73afb1c35b82d264346c2ae2b5981e`。

| 项目 | 估算/已知值 | 解释 |
|---|---|---|
| Base权重下载 | 7,509,245,952字节（7.51GB） | BF16，按2字节/参数推导约3.7546B独立序列化参数 |
| Released权重下载 | **16,263,151,616字节（16.26GB）** | config为FP32，metadata约4.0658B序列化参数；比base多一个`lm_head.weight`键，不把文件参数数目当运行时独立trainable参数数目 |
| 两份权重同时存储 | 23.77GB + tokenizer/265张card | 不默认release也是7.5GB |
| 单个BF16运行模型 | 名义7.51GB权重，保守按8.13GB计 | 模型实际load和tie行为须D阶段确认 |
| actor FP32参数 | 约15.02GB | 官方FSDP参数存储FP32，BF16混合计算另算 |
| critic FP32参数 | 约15.02GB + value head | 不是一个很小的独立MLP，主干本身也是VLM |
| actor+critic Adam moments | 约60.07GB CPU/GPU总存储 | 每参数两份FP32状态；CPU offload减少GPU驻留，不消除RAM和checkpoint体积 |
| 两模型+完整Adam checkpoint | 名义90.11GB，可能更大 | 尚未计权重下载、环境、HF重复导出等；**即使100GB也不能原样全部保留** |
| 两模型weight-only checkpoint | 名义30.04GB，保守32.53GB | 无optimizer，无额外HF完整模型副本；保留extra、dataloader与skill快照 |
| BF16 KV cache | 每token每sequence36,864字节 | 2(K/V)×36层×2KV heads×128head_dim×2bytes |
| 长度9216、并行1个sequence | 约0.316GiB KV | 不含vision encoder、allocator、graph和引擎预分配；TP1每个DP副本单独有模型和KV |
| 9000tokens全序列logits | BF16约2.55GiB；FP32约5.09GiB | vocab151936；官方critic patch可能构造/升精度logits，额外峰值不能忽略 |

显存估算不能将3张卡简单相加：TP1的rollout每个副本都加载完整模型；FSDP分片训练参数、optimizer可offload，且有层级all-gather、activation、梯度、logits临时量。两个训练主干+一个rollout副本在单卡的纯参数上界约35–38GB；CPU offload在不同阶段降低驻留，但首轮要预留额外20GB以上给activations/logits/引擎。**单卡72GB有合理smoke余量，但实际峰值尚未测量**。建议首次目标<60GiB；显存高于估计时记录阶段与错误，不自动改算法。

## 4. 四级可行性结论

| 等级 | 结论 | 依据与条件 |
|---|---|---|
| A evaluation only | 硬件适合，软件验证待C/D | 单卡BF16 3.75B，batch1、最多512tokens/turn有充足权重/KV余量 |
| B component smoke test | 适合 | CPU/RAM/真实渲染及本地单模型容量足够，实际完整组件测试尚未执行 |
| C reduced PPO reproduction | **算力可行，执行有前置条件** | 全参actor+critic、FP32参数/BF16计算、offload和tiny batch可规划；FA/SGLang、NCCL、reward一致性和checkpoint配对仍需解决 |
| D full paper-scale reproduction | **当前不成立且不授权启动** | 非H100、论文2/4卡矛盾、最多3指定卡、multi-GPU未通过、完整checkpoint预算不合；无吞吐证据，不能保证300步≤24h |

FSDP：架构/内存条件适合，但实际多卡通信目前FAIL；不是已经验证FSDP。单卡PPO也需验证world-size1进程组、critic/FA2与rollout兼容。

## 5. 可行候选配置（尚未运行）

配置为规划记录 `evidence/hardware/proposed_config.json`，不是通过D/E后的可启动smoke YAML，不冒充阶段F实现。

### 本地验证

- GPU4单卡，模型顺序加载/卸载，BF16、Transformers直接推理、batch1。
- 每组20个固定seed10001–10020；Base无库 / Release无库 / Release+发布库，三组一致seed。不用于声称复现论文success rate或调参。
- 保留原生processor和card；每turn≤512tokens、最多5turns、5actions。context候选9216；真实图像token与prompt长度须测量，不能为通过而裁剪关键视觉技能。
- 不启服务/付费API；如未来必须服务，仅loopback。正式GPU进程及子进程必须使用中性入口/参数并检查完整ps/nvidia-smi。

### 首次全参PPO smoke候选

- 优先**GPU4单卡**（通信问题未解决前），2个training steps，train batch4，actor/critic mini-batch2，micro-batch每卡1，n=1。world1 NCCL仍需验证，不默认为通过。
- 固定训练采样seed42，训练seed域1–10000，val仅2个固定held-out seeds10001/10002；held-out不入库、不挑seed保证成功。
- prompt5000、response4000、per-turn512、max_model_len9216，保留官方上限，先减batch不裁剪关键轨迹。
- actor/critic继续全参，原LR1e−6/1e−5、GAE1/1、clip .2/.5；FP32参数、BF16计算、activation checkpointing、param/optimizer CPU offload；LoRA=0、cold=false、distill=self、no reference。
- rollout TP1、并行sequence1、agent worker1、memory utilization .25、max_num_batched_tokens10000，eager/free-cache。内存比例是引擎预算，是否能容纳graph/模型必须实际初始化验证。
- console only，wandb disabled，HF upload=null；每step存一次，只保留一个checkpoint；actor/critic均 `save_contents=load_contents=[model,extra]`，不导出另一份完整HF权重。
- **明确偏差**：此smoke checkpoint可以恢复policy/critic权重、extra和对应skill库，但不保存Adam moments，不能声称完整optimizer-continuous训练恢复。PPO运行内仍使用Adam并验证梯度/参数更新，不改成SFT或删critic。保留optimizer状态摘要作为更新证据。
- 若两个steps没有符合质量门的成功轨迹，skill update验收仍为PARTIAL；不注入solver cold-start、不挑held-out、更不将失败改写为PASS。

### 多卡候选（必须先通信通过）

- 两卡首选4/5（同NUMA）：batch4、mini2、micro1，TP1。
- 三卡3/4/5：batch6、mini3、micro1，TP1。不能将batch4/8或mini2/4直接复制到3个FSDP ranks；官方静态整除检查及world-size归一化会不合法。
- 两卡/三卡是保留的后续候选，不自动尝试全部方案或扩大实验窗口。

## 6. 存储、环境和独立约束

模型下载23.77GB；独立环境与依赖暂给15GB，新增缓存5GB，weight-only actor/critic checkpoint保守32.53GB，图像/日志/库与临时余量5GB，总规划约81.30GB。以100GB为hard cap，**85GB时停止新增工作并终止本次写入进程组**，留出下载分片/清理/状态保存余量；安装与下载实际体积必须持续低成本统计。

历史数据和其他环境不计入可清理对象；新环境用中性路径`/data/local/envs/z0`，项目源码/模型/日志在`/data/local/ViSkill`，工具`/data/local/tools/z0`，临时文件`/data/local/tmp/z0`，四处新增均计入预算。不覆盖任何已有路径；不使用根/home缓存。无预先可复用的base模型结论：仅在两个已检查cache未发现，不声称全服务器不存在。

服务器无PATH中的conda/uv/python3.10；采用公开micromamba创建独立Python3.10前缀，工具与包缓存也记账；这是安装工具适配，不更改训练算法。不改系统Python、驱动、CUDA或已有环境。系统nvcc13.2与拟用torch CUDA12.8不同，不能无检查地用其编译FA；先可用SDPA完成本地验证，PPO FA/SGLang单独验证。不会执行install_render.sh。

## 7. 当前交付与下一步

阶段B硬件事实与可行性审计完成；单卡VLM/PPO和多卡FSDP尚非动态PASS。新授权已允许继续C→D→E→F，严格保留门槛。当前唯一优先动作是独立环境安装与固定revision权重下载；setup自身为CPU/网络任务，不分配GPU，不自动训练。

半小时唯一监测：`viskill`（绑定当前会话）。setup独立supervisor负责6小时硬超时和85GB写入停止线，实验未来单次仍受24小时上限；监测不替代硬约束，安装失败后修复需沿用原始计时与失败记录。后续状态以远程receipts、退出标记和实际产物为准。
