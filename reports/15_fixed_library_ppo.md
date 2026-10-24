# 固定真实库PPO：权重×上下文的受控机制实验

状态：第一模型调用前冻结、CPU预检通过，A已完整完成，B已启动训练前基线交叉评价。依据第14报告的四条件投入门槛通过，自主推进此有限实验；它不是旧bootstrap的续训，也不是原论文空库在线自举。

## 问题、对照与预算

两臂从原b0重新初始化。A普通全参数PPO，训练时空库；B相同PPO，训练时使用原bootstrap B实际产出的最终5卡冻结库。两臂均无在线准入、无utility更新，current-policy self-distillation只保留旁路，质量/novelty/guard不变，技能奖励权重0。固定历史库属于明确机制变体，不冒充训练cold-start/self在线自举通过。

每臂4批×32＝128训练，mini8/micro1/n1，标准actor1e-6、critic1e-5和原clip/完整视觉塔/全参数设置，不搜索。每批固定8个BFS1、8个实际skill_4适用Up×3、8个其他BFS2–3、8个BFS≥4，保证B每批8次三步卡实际参与更新；实际processed图像审计与batch计数必须验证，而非只数候选。A同图同顺序，不增强对照。

两臂各在theta0、theta2、theta4，均以空库和同一冻结5卡库，greedy评价自然已开发Dev64与全新条件Dev32，共每臂576评价局。满额每臂704局、两臂1408局，最多7296模型请求（每臂3648，训练每局最多5动作请求+1蒸馏，评价每局5动作请求）；CPU显式fixture不混入这些分母。不保存新权重，所有交叉评价在trainer存活时完成，退出后不能补测/续训，不验证独立恢复，完整F继续PARTIAL。

theta2完整交叉后，如果训练上下文的自然多步成功或质量相对同上下文theta0净退步≥4/53，该臂按冻结规则在2批停止。theta2交叉已完整保留，不能挑最好时间点替代预定theta4，也不重放完成的局。另一臂按原协议独立运行；任一臂未到4批，最终投入门槛不通过。

## 数据冻结

训练固定候选90001–106384，各类别取升序前32个实际新状态；然后每批每类8个。排除3681个历史实际棋盘/位置（包括预留Test与前两次条件诊断全部已筛选地图）。训练扫描前3796个候选形成四类各32；新条件Dev固定池110001–126384，额外排除全部训练候选前缀，扫描前3718得到32张新适用图。未看模型输出、未放松适用门槛，没有按模型成功选题。旧自然Dev64继续是开发集，不再称封存测试；两套Dev分别报告，不能合成总体自然成功率。原Test256无模型请求。

实际PilotDataset/StatefulDataLoader检查128训练正好4批、96Dev正好12个8局批，seed顺序和sentinel排除通过。全部精确状态、候选分母和seed表见冻结协议`configs/smoke/fixed_library_ppo_v1.json`与`docs/evidence/components/fixed_library_ppo_v1/data_manifest.json`。

## 参数收益的主比较

主终点只用同一个冻结库上下文下的新条件Dev32：B4−B0及B4−A4，均报告逐图成功和路径质量改善/退步与不确定性。另完整报告各权重空库结果，识别无卡迁移；不要求所有有卡学习必须内化，也不把即时卡片作用当参数学习。

继续扩大投入须同时满足：两臂完整4批、所有实际输入/数值/critic配对检查通过；B自身和B相对A，在同库条件Dev的成功与质量均净增至少4/32且每项退步至多2；B自然多步相对自身退步<4。此为小预算筛查，不是单seed稳定性能证明。未过则关闭本协议，不加第五批或挑seed追阳性。自然多步保护线优先于条件任务收益。

## 初始化与真实数值检查

使用限定于新worker的critic初始化torch seed20261024，在fork_rng内生成，保存rank0完整小value-head数值，B初始化时必须与A记录完全一致；相同原b0作为actor/critic骨干。没有为此复制全FP32权重，也不声称仅凭小head和相同路径证明整模型逐字节identity。

run seed20261024、data seed42、SGLang server random_seed42和地图seed分别记录。安装的SGLang SamplingParams不支持请求级sampling_seed，故不伪造严格采样配对；其greedy分支为argmax，不使用非贪婪采样抽样，但并发replica/请求分配仍限制随机流的精确配对。本轮只是一训练seed机制pilot，不宣称覆盖训练seed变异。

首个真实训练batch，在两GPU每rank至多8个真实样本，额外做相同pre-update权重的actor eval old_logprob与train-mode forward比较，RNG保存恢复；记录实际多图输入、全部误差和额外forward时间。BF16两个执行路径的冻结误差门为max≤.05 nats、mean≤.005，低于PPO .2 clip尺度；这是第一新模型调用前按数值精度设置的容差，不是事后放宽。不是SGLang-vs-actor logprob证明。真实gamma=lambda=1的terminal aggregate reward，所有generated-token return与整局reward误差≤.001，reward只落generated mask，values/returns/advantages/old_logprob finite；记录至多16条实际reward分解。原actor/critic损失mask和optimizer不改，沿用delta/sync审计。

CPU实际processor/Gymloop用4个显式scripted输出并发交叉empty/cards，确认A也可评价cards、实例间provider不泄漏、Dev无蒸馏、0模型请求。两臂配置validate通过，现有5项bootstrap phase/mask测试通过；这不是实际新PPO objective通过，首批真实数值门仍须运行时通过。

## 守卫与交付边界

准备起点2026-10-24 20:03:02 Asia/Shanghai；含全部准备最多2小时30分，pair硬截止22:33:02，单臂自开始最多60分钟，原campaign截止2026-10-25 00:19:41不变。新增总2GB、1.8GB写停止预留.2GB，另受85GB项目保守线约束；远程独立监督器每5秒检查并仅清理本token进程。旧模型/检查点不删，禁止新权重保存/外部推理API/遥测。仅物理GPU3/5两卡，启动前检查各空余≥44000MiB，不动他人任务；所有进程argv中性并启动后检查。

任何init/数值/输入/资源故障先保存原始错误和已完成调用/更新；不从头重放或重置预算。只保留现有唯一30分钟跟进，健康静默。未完成不称完成；公开推送继续按本项目的明确发布请求边界执行。

启动回执：supervisor286374，A自2026-10-24 20:23:21开始、单臂硬截止21:23:21；pair硬截止22:33:02。baseline68532609024、write_stop70332609024 bytes；CPU预先写入计入独立证据，另有.2GB预留。当前初始化head记录已产生，24自身进程argv违规0，GPU显示中性；尚无新模型请求或实际首批数值门通过结论，不能把启动称为训练成功。

启动后简短确认：A已实际完成71条基线交叉episode、253原生模型调用，更新0、无EXIT；正在按预定theta0评价，不是卡住或已得到学习结果。


2026-10-24 20:46短确认：A已完成2/4批（64训练局），正在theta2完整交叉评价；累计350局、1268模型调用，监督器存活且无EXIT。A首批真实32行returns/finite/reward mask门已通过，最大return误差5.327165126800537e-7（门.001）；两rank共16实际样本old actor eval与同权重train forward的最大/平均logprob差均0。两批共32条optimizer审计（actor16/critic16）全部finite且采样delta非零。此处多图主要是多轮环境观察；B带卡首批尚未运行，不能把A通过当B已通过，也不是SGLang-vs-actor概率一致性证明。没有新的学习收益结论。A原始3份数值审计已归档docs/evidence/components/fixed_library_ppo_v1/A/numerical。


A已于21:08完成全部4批与704局/2573调用，B21:08:15按原配置启动。A完整结果与后续状态见[16](16_fixed_library_results.md)，不再将A中期评价视为当前运行。
