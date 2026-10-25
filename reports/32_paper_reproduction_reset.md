# 回到官方方法复现：清理旧权重，冻结官方256图发布模型核对

用户最新优先级是先较准确复现原论文，并明确允许删除不需要的大文件。已读用户提供的分析：其中对结果边界、行为概率、精确输入和故障成本的提醒保留；轨迹记忆、监督正对照等属于新方法研究，当前不启动。旧缩减诊断结论不作为原论文性能复现失败的证据。

核对来源：[论文v1](https://arxiv.org/html/2610.12403v1)、[官方代码](https://github.com/ZJU-REAL/ViSkill/tree/7acaff880871b8d898d91c515c151080eabcba20)、[发布模型说明](https://huggingface.co/hongxingli/ViSkill-FrozenLake/blob/edafb2b71f73afb1c35b82d264346c2ae2b5981e/README.md)。

原12小时窗口已按原时钟关闭，原因12_hour_campaign_deadline，原截止11:38:27、守卫结束11:39:02；无GPU实验延续到下一窗口。当前是新用户请求下的独立复现阶段，不重置任何旧实验时钟。旧窗口新增约15.88GB且低于50GB界，完整关闭记录见[receipt](../docs/evidence/reproduction/reset_20261025/autonomous_20261024T153827_public_closure.json)。

## 1. 新的源码与方法边界

官方GitHub HEAD仍为7acaff880871b8d898d91c515c151080eabcba20，已通过显式代理核实。在独立的干净工作树上准备，源码提交4039c7680cd9a2be16e6e2b5bf13e5433061f409；运行树从该官方commit归档产生，不复制私有Git历史。除了已观察到的并发RGB关闭死锁所需生命周期修复，只新增配置、记录器及守卫，没有采用此前诊断方法分支。见[源清单](../docs/reproduction/source_manifest.json)与[运行修复patch](../docs/reproduction/official_runtime.patch)。

保持原始free-text自蒸馏、几何检索、发布库和skill_guided奖励0.8；无directional合同/额外严格guard、准确动力学改写、Up3筛选课程、零值头或答案mask。原始提示、动作解析、环境transition及完整官方数据不改。RGB close修复只避免关闭整个进程的SDL，不改轨迹动作、图像绘制或奖励；复用实际8环境生命周期fixture通过，单独计1次fixture env.step/2个动作，0模型调用。

先执行官方发布模型hongxingli/ViSkill-FrozenLake@edafb2b71f73afb1c35b82d264346c2ae2b5981e及其265张原技能库，在官方验证集10001..10256上完整核对。train仍按官方1..10000留档，不实际训练；val一次256条、8 agent workers、greedy T0、最多5回合/回合512tokens/最多5动作，prompt5000/response4000。没有历史或按结果筛选，也不把已知的条件16图成功当本次结论。

## 2. 必须明示的差异和未完成项

| 项目 | 原资料/当前选择 |
|---|---|
| 性能参照 | 论文FrozenLake非cold-start主结果0.84；发布model card未标明cold-start变体或训练seed，不能保证其精确对应某一主表行 |
| 本阶段目标 | 发布模型性能核对；不是从b0重新训练成功，不声称已复现完整论文 |
| GPU | 官方脚本4卡，附录另列2卡；本次用授权内物理3/5两卡，保持完整val batch256。GPU型号为RTX PRO 5000 Blackwell，不冒充H100 |
| 注意力运行时 | 现有已验证Blackwell环境采用text Triton/vision SDPA；保留实际BF16路径，不能要求所有后端逐位相同 |
| 未使用critic | val_only阶段用selector跳过无用critic，无优势或GRPO/PPO更新；原训练配置单独完整留档 |
| 奖励忠实性 | 第一条线明确复现released-code的动作前缀奖励。论文附录的状态进度定义与之不同，保留原审计，不静默混合两种算法 |
| 训练预算 | 原文300步、脚本301；后继必须实测原batch128/mini32配置吞吐、checkpoint峰值和单实验24h/新增50GB约束，不能提前承诺已跑满300步，也不拆名绕过时钟 |

评价使用官方实际原生agent与production hybrid同步路径；记录器只保存收到/返回的原token、图像、episode、加载信息，不改变原loop返回。原validation路径可能进行完成后的self-distill，保持其行为并分开计数；不运行入库/utility回调，库JSON值必须不变。输出完整成功率、与0.84差值及Wilson95区间，不在结果后修改容差或把区间覆盖0.84称为等效证明。

## 3. 用户授权的定向清理

已核实无活动消费者，锁定本项目根和5个实际文件后，删除旧directional smoke的4个FP32模型分片，以及已由私有BF16差分完成实际恢复核对的Z冗余完整权重。删除实际allocated40,037,646,336 bytes，总四根从约84.37GB降至44,332,302,336 bytes。未做hash或逐字节比对。

保留原始b0、官方r0及完整发布库、O完整快照、Z/A私有差分、原输入和所有报告/source/config/numeric/失败。旧smoke的metadata/pair记录仍在，但其4个模型分片退休后已不能按原路径恢复权重；不能再声称32.56GB旧恢复checkpoint完整保留。Z的私有推理恢复能力保留，非FP32/optimizer续跑。见[精确清理清单与完成记录](../docs/evidence/reproduction/reset_20261025/cleanup_20261025_paper_priority.json)。本次授权不扩大到别的项目或删除原始模型。

## 4. 调用前冻结与当前状态

协议[released_frozenlake_eval_v1](../configs/reproduction/released_frozenlake_eval_v1.json)：准备起点11:49:20，含准备3小时原截止14:49:20保持；最多256局、1280次rollout dispatch及256次完成后self-distill、总1536，0optimizer/入库/新权重。新增硬1GB/900MB写停，含新源码和所有缓存、图像、日志；独立5秒空间/原钟/own-token+ctime/argv守卫，失败保留实际调用，不重放已经dispatch的评价。

CPU已通过：实际Hydra配置、官方10000/256 seed全集、模块均来自干净树、无原诊断扩展、记录器返回对象不变及调用前上限、正确Worker MRO、265张发布库、8环境生命周期。0模型调用，CPU fixture不混入256性能分母。见[CPU回执](../docs/evidence/reproduction/reset_20261025/cpu_check.json)。

本文为调用前协议，尚无本次256图性能结果。后续只有实际COMPLETE/退出/全256结局齐备才汇总；不以模型初始化或工程检查替代复现结果。后继完整训练在发布模型核对之后推进，仍沿用同一唯一监测，不开展新的改进支线。


## 启动回执（尚无性能结果）

12:09:06已启动独立supervisor2537307，原14:49:20含准备截止不变。CPU/原设置检查通过后，使用干净执行提交b856d7ae941d2df5abf10d3b7efdd0943093d863；新增明确server context9000与官方wrapper的5000+4000上限相同，未改请求长度。初始GPU3/5剩余69759/66207MiB，四根占用44,355,239,936 bytes，写停45,232,302,336。首次检查进程存活、日志可读、无立即失败，实际加载官方r0四个shards，24个own进程argv违规0。该观察只确认加载阶段健康，不是256图完成或成功率。见[启动回执](../docs/evidence/reproduction/reset_20261025/released_frozenlake_eval_v1_public_start.json)。


## 初始化失败与一次调用前修复

首次worker于12:11:58退出，错误为`ActorHandle has no attribute sleep`。问题来自新增观察器错误继承了同步ActorRolloutRefWorker，而官方async配置选择AsyncActorRolloutRefWorker。这是本次实现错误；先前CPU检查只覆盖Worker MRO，未覆盖异步RPC，故未提前发现。模型权重实际加载成功，但尚未进入evaluation、没有任务dispatch、episode、自蒸馏或optimizer更新；GPU初始化成本不为零，不能记成零计算或论文性能失败。

原始输出目录、完整日志和退出回执保留。已确认无本次残留活动进程，CPU检查实际配置async、继承官方AsyncActorRolloutRefWorker、sleep/wake_up/generate接口及Ray Worker MRO全部通过。只允许一次任务调用前修复；原准备起点、14:49:20截止、总存储口径及完整256图设置不变。后续再失败则保留证据关闭，不自动重放。见[失败与修复审计](../docs/evidence/reproduction/reset_20261025/initialization_failure_audit.json)。

12:20:37已在同一原钟下启动第2次初始化，执行代码冻结d7652143f446c7445d78855163fb7eeb9fd13757。启动检查确认官方异步sleep/wake_up/generate接口已实际绑定，24个own进程命令行违规0；尚无256图性能结论。见[第2次启动回执](../docs/evidence/reproduction/reset_20261025/released_frozenlake_eval_v1_retry_public_start.json)。此处补充保留前述初始启动事实，不覆盖原失败。


结果汇总器另完成一次256条纯合成CPU fixture，覆盖比例计算、完整请求导出、失败成本合并和token剥离；0模型/环境/optimizer调用，不计入真实性能分母。
