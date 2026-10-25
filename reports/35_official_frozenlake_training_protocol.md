# FrozenLake从基础模型训练：官方配置调用前协议

用户要求先较准确复现原论文，并在2026-10-25 13:02:07明确同意：**仅本次官方复现新增存储<150GB、四根合计210GB，24小时上限不变。** [报告34](34_official_training_storage_feasibility.md)的105.13GB估计及原50GB预算冲突保留；新授权不追改其他实验上限。[报告33](33_official_released_frozenlake_check.md)的216/256是发布模型核对，此处是不同的从基础模型训练阶段。

## 数据和研究目标

任务为FrozenLake视觉导航，使用官方完整训练seed1..10000和验证10001..10256。原AgenticDataset扩展已实际验证10000/256个seed全集；按官方DataLoader/sampler训练，不按历史胜负、任务类别或外部答案筛选。初始化为现有原始Qwen/Qwen2.5-VL-3B-Instruct b0，**新的空在线技能库**，不用发布r0模型/265张库、旧5卡或私有学习差分作为起点。

本路线明确CODE_AS_RELEASED：batch128、mini32，actor lr1e-6、critic lr1e-5，原GAE/PPO及完整生成response mask，原FP32 AdamW状态，原free-text self-distill、retrieval minscore0.8、skill_guided0.8动作前缀奖励和在线库上限10000，非cold-start。保持原prompt5000/response4000、每回合512tokens/最多5动作/每局最多5回合，训练T1、验证T0。

主目标为脚本的301个global训练batch，主结果取最终step301的完整256图验证。另列论文名义step300及所有中间验证，不用验证最优checkpoint替代主结果。原验证频率20、训练前验证、最终301验证，最多17次、4352个验证episode。训练最多38528个episode；rollout dispatch上限214400、完成后self-distill上限42880，总257280。内部后端初始化/可能重试不冒充外部dispatch，故障和成本单独保留。

本次单训练seed42，不声称跨训练seed复现或统计等效；与论文0.84作数值参照。脚本301vs论文300、脚本4卡vs附录2卡、原代码动作前缀奖励vs论文状态进度定义的差异继续明示，未静默混合算法。

## 必要执行差异

- 使用授权物理GPU3/5两卡；保留原全局batch和mini-batch，不通过缩小训练问题省预算。
- SGLang推理使用已验证Blackwell的文本Triton/视觉SDPA、context9000；训练保留原HF/FlashAttention2和fused kernels路径。现有环境复用，不新下载模型/依赖。
- console和本地结构化记录，不向wandb/HF等外部服务上传。原训练参数和原生fit/update算法保持。
- 唯一上游核心补丁仍为已验证的FrozenLake RGB close生命周期修复，不修改transition、动作解析、奖励或prompt。
- 观察器保存全部训练/验证结局、实际请求prompt IDs/原始输出/output IDs/采样/长度/时间；完整输入PNG仅保存首次遇到的固定train seed1和val seed10001 episode，不按结果选择。原生loop返回对象和server.generate参数保持不变；不声称保存了内部processed多模态训练tensor。

## 保存和24小时预算

准备起点固定13:02:07.566，含本次准备、故障和任何续跑的硬截止为**2026-10-26 13:02:07.566**。提前90分钟发协作停止请求，在当前完整原生batch边界保存并退出；最终硬超时独立于Codex/SSH执行，不为重试或阶段改名重置。

四根基线44,402,794,496 bytes，实际记录于13:03:07。授权新增严格<150,000,000,000 bytes，四根总210,000,000,000 bytes取严。更保守地在新增130GB时写停，留20GB至硬界；每次完整保存前用“非checkpoint新增+120GB完整状态预留”复核，不仅依赖半小时心跳或轮询事后截停。挂载、剩余容量、write/read探针及GPU显存由独立supervisor在启动时检查。

原actor保存model/hf_model/optimizer/extra，critic保存model/optimizer/extra；原每100步及最终保存，原各保留一代。额外在首个完整batch后保存一次，实测原状态峰值和保存耗时，不改变任何更新、奖励或数据选择。完整在线库另按同一训练步快照配对，保留原dataloader/RNG/scheduler状态；核验原生保存返回和必要文件集，不默认hash或重载全模型。约105.13GB只是假定完整dense Adam状态的tensor估计，真实序列化/别名/padding/日志等峰值尚待实测。

保留官方单代role轮换：它会在新role保存前删除该role上一代，故不声称断电期间双代原子恢复。遇到预算/时钟/非有限loss或grad/接口失败时保留失败、实际调用和最近完成的恢复点；不得盲目重采样、静默换optimizer/精度/LoRA或延长24小时。若未达到301步，报告预算受限或工程未完成，不把其当作算法否定结论。

## 已完成CPU检查和当前边界

CPU_READY于13:22:13通过：完整原Hydra配置与GAE/critic、原数据seed全集、async actor必要RPC及actor/critic共同Worker MRO、真实父类128条_get_gen_batch、记录器参数和返回对象不变、调用前次数上限、当前安装版本HF导出构造器meta可用。无GPU、模型forward/backward、原生env.step或optimizer更新。本次未重做报告34那个标量Adam fixture。见[CPU回执](../docs/evidence/reproduction/official_frozenlake_ppo_v1/cpu_check.json)。

CPU通过不证明完整batch显存、FSDP backward、实际保存峰值或301步可在24小时内完成。首个原batch完成后才给实测耗时及带限制的剩余时间估计，发布模型42秒验证不用于外推。代码、协议及预算在启动前冻结并公开，后续实际启动/完整结果追加保留。当前尚无本阶段训练性能结果。


干净执行源码冻结提交：`a3e18a54cfc35180193954fae1536c52da733b05`。该记录在GPU调用前加入；后续只追加实际状态，不把CPU/初始化当作训练成功。
