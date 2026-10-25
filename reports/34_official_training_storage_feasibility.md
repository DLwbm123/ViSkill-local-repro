# 官方完整训练的保存预算核对：当前50GB上限不足

发布模型已完成[官方256图核对](33_official_released_frozenlake_check.md)。下一阶段恢复基础模型、空技能库及官方在线PPO训练配置；在启动前核对发现，**保持官方完整checkpoint时，现行单实验新增<50GB及四根85GB保守线都不够**。本轮没有启动GPU训练、重新评估或新方法。

## 已恢复并检查的训练配置

从固定官方脚本和Hydra默认项恢复：train seed1..10000、val10001..10256；batch128、mini32；GAE、完整生成区域PPO；actor lr1e-6、critic lr1e-5；原self-distill、原skill_guided0.8、从空库在线积累、非cold-start。critic已恢复，不沿用验证阶段跳过critic的selector。训练目标按CODE_AS_RELEASED为301步，保存每100步，验证每20步且训练前验证。max_actor_ckpt_to_keep及max_critic_ckpt_to_keep原本均为1，不是无限保留。

现有授权下拟用2卡，保留完整全局batch；与脚本4卡、论文附录2卡的差异继续明示。GPU训练显存和吞吐尚未测量，发布模型41.94秒greedy验证不能外推基础模型300/301步训练。参数、奖励和在线机制均未为适配资源而改动。

## CPU证据及数值边界

仅读取现有b0的safetensors header，并在meta设备构造原actor和原TRL critic架构，不加载实际参数、不执行forward/backward。实际生产worker会把actor及critic完整模型转为FP32；按相同dtype计数。一个标量CPU AdamW fixture执行1步，仅确认原配置exp_avg和exp_avg_sq均为FP32，不是模型训练。

| 项目 | 数值 |
|---|---:|
| actor独立且可训练参数 | 3,754,622,976 |
| critic独立且可训练参数 | 3,754,625,025 |
| critic value head | 2,049 |
| 每个模型decoder layers参数 | 2,774,771,712 |
| actor+critic FP32模型有效载荷 | 30,036,992,004 bytes |
| 两模型完整FP32 Adam moments有效载荷 | 60,073,984,008 bytes |
| 模型+仅decoder moments的保守核对 | 74,433,339,396 bytes |
| 模型+完整dense moments | 90,110,976,012 bytes |
| 官方actor重复HF导出的有效载荷 | 15,018,491,904 bytes |
| 保持原保存内容的总有效载荷 | 105,129,467,916 bytes |
| 当前四根85GB线下剩余空间（上次闭环口径） | 40,597,246,464 bytes |

这是根据实际架构和FP32 AdamW配置计算的**未压缩tensor有效载荷**，不是已实写checkpoint大小。完整dense估计假定Adam状态已为相应参数建立；仅decoder moments的保守口径不计视觉/embedding/value-head moments。实际序列化开销、重复别名、FSDP padding、RNG、dataloader、技能库、日志、缓存和临时文件另计，压缩收益没有预先假定。FSDP把文件拆到不同rank，不降低同一服务器总量。

actor的官方save_contents为model/hf_model/optimizer/extra，critic为model/optimizer/extra。即使取消重复HF导出，约90.11GB核心状态仍超过50GB；多删历史文件只能改变总空间，不能改变“本实验新增<50GB”。不能把单个rank文件较小当作整台服务器合规，也不能默默丢optimizer、换低精度optimizer或LoRA来冒充原训练配置。

CPU准备从12:48:25.955起，原10分钟/2MB界内于12:52:09.531完成；0 GPU、0模型forward/backward、0原生环境step、0真实actor/critic optimizer更新，仅1个标量CPU fixture step。中间两次CPU构造适配失败均保留：旧AutoModelForVision2Seq.from_config只接受config；meta构造需补上与生产worker相同的完整FP32 cast。它们没有进入模型计算或GPU阶段，未重置本轮CPU时钟。见[完整数值与设置](../docs/evidence/reproduction/official_train_checkpoint_cpu_v1/result.json)。

## 可审查的资源方案

当前完整训练保持未启动，数据、模型和旧结果受保护。建议只为这一次官方训练作显式存储例外：

- 每实验新增实际存储严格小于150,000,000,000 bytes；四根合计保守线210,000,000,000 bytes。保持原24小时硬总时长和授权GPU范围，重试/续跑/子任务共用原钟。
- 复用现有b0、数据和缓存，保留原模型/optimizer/scheduler/RNG/dataloader保存内容及官方单代checkpoint轮换。约105.13GB数值载荷之外约44.87GB用于实际序列化、别名、库/日志/缓存与余量；这不是已经验证的峰值承诺。
- 启动前冻结完整运行方案，实测首个原batch128训练周期的吞吐和可恢复保存峰值。首阶段属于同一原始24小时预算，不拆名续跑。实际写入前复核预算，140GB新增写停、150GB硬界及总210GB线取严，独立守卫不依赖半小时心跳。
- 若存储峰值或原配置吞吐无法在预算内完成，则保留失败、成本和最小可恢复状态后结束，不把预算中止当成算法性能结论，也不继续延长实验。

原单代保存实现会在写新一代的对应角色前删除上一代，因此断电/保存故障期间不能保证仍留有一组完整旧代；本方案保留这一官方行为，不额外声称具备双代原子恢复。若需要先完整写好新一代再退休旧一代，需另行审核约双倍checkpoint峰值和保存机制，150GB方案不覆盖该要求。

若保持现行50GB/85GB约束，本次约105GB完整原保存方案不能启动。无需为此重新跑发布模型、改算法或开展无关支线；下一步取决于用户是否为本次官方复现修改存储预算。CPU核对本身不证明24小时训练可行或完整论文已复现。
