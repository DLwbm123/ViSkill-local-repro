# 官方发布模型的完整256图核对：216/256

**官方发布模型及其265张技能库，在官方完整验证seed10001..10256上成功216/256，成功率84.375%。** 这与论文报告的84%数值接近，绝对差+0.375个百分点。当前完成的是发布模型核对；尚未从基础模型重新训练，不能据此声称完整训练复现、统计等效或恢复论文训练seed。

调用前的[协议、执行差异和清理边界](32_paper_reproduction_reset.md)保留；官方源码固定7acaff880871b8d898d91c515c151080eabcba20，执行代码修复提交d7652143f446c7445d78855163fb7eeb9fd13757。官方模型revision edafb2b71f73afb1c35b82d264346c2ae2b5981e，原始265张库，无诊断变体、手选子集、额外guard、答案mask或零值头。

## 完整结果及指标口径

| 指标 | 本次完整结果 |
|---|---:|
| 官方验证图 | 256/256，原seed全集，每图一次 |
| 成功/失败 | 216/40 |
| 成功率traj_success | 84.375% |
| 与论文0.84参考值之差 | +0.00375 |
| 描述性名义binomial Wilson95区间 | [0.794229, 0.883108] |
| 原代码平均reward | 0.918750，含原奖励塑形，不作为成功率 |
| 平均/最大模型回合 | 1.109375 / 5 |
| rollout dispatch | 284 |
| 完成后的self-distill dispatch | 1 |
| optimizer更新 / 入库 | 0 / 0 |
| 技能库JSON值 | 全程不变 |

区间只作单次地图样本的描述性参考，不表示跨训练seed不确定性，也不证明统计等效。发布model card未明确cold-start变体或训练seed来源，不能保证checkpoint精确对应论文某一表行。无额外重复评估、重采样、按结果挑seed或事后容差。

全部285个实际请求为T0。rollout输出7649 tokens，self-distill输出20 tokens；rollout原prompt长度合计158639 tokens。计数指agent dispatch；后端初始化与可能的内部实现调用不是零成本，也不冒充外部dispatch。主验证阶段12:21:40.584至12:22:22.528，约41.94秒；supervisor于12:22:30.296关闭。发布模型的短greedy轨迹吞吐不能直接外推基础模型在线PPO训练耗时。

## 加载、故障和资源闭环

实际HF加载missing/unexpected/mismatched/error全部为空，embedding与output weight实际共享。全256 episode和请求数一致，284 rollout等于全部num_turns之和，1次self-distill分开计数，库JSON值未变，COMPLETE和EXIT0齐备，命令行违规0，闭环无本轮活动残留进程。

首次初始化因新增观察器继承同步工作类、缺少sleep接口而失败；没有任务请求或episode，但有模型加载/GPU初始化成本。已明确记录实现错误，保留完整原日志、输出和退出回执。只在0任务dispatch条件下修复为官方async工作类，沿用原准备11:49:20、原14:49:20截止及原存储口径；没有重放已采样地图。原失败详情见[审计](../docs/evidence/reproduction/released_frozenlake_eval_v1/initialization_failure_audit.json)。

清理后四根基线44,332,302,336 bytes；闭环44,402,753,536 bytes，本阶段含两次初始化累计新增70,451,200 bytes，低于900MB写停和1GB阶段硬界。41.94秒为实际主验证阶段时间，不含前置准备和首次故障；从原准备至第2次退出约33分10秒，未重置时钟。没有新增权重；原有被保留模型/差分/数据仍受保护。

## 可公开证据与未完成项

[结果JSON](../docs/evidence/reproduction/released_frozenlake_eval_v1/result.json)含全部256结局、指标、加载信息及原失败成本；[episodes](../docs/evidence/reproduction/released_frozenlake_eval_v1/episodes)与[requests](../docs/evidence/reproduction/released_frozenlake_eval_v1/requests)保留全部成功和失败、原始生成回复、输出token IDs、采样参数和长度。只导出事前固定seed10001/10002的首请求输入示例图，不按结果选择。批量原始prompt IDs和输入图像、完整运行日志留在受控存储，模型权重和私有Git历史未公开。CPU汇总/export fixture使用256条合成记录，0模型/环境/optimizer调用，未混入真实性能分母。

后续按用户优先级准备基础模型的官方训练配置：原train1..10000/val10001..10256、batch128/mini32、actor lr1e-6/critic lr1e-5、完整生成区域GAE/PPO、原self-distill、原在线库和原reward。先核实完整可恢复checkpoint峰值与实际训练吞吐是否满足单实验新增<50GB/总≤24h；不得靠丢恢复状态或拆名续跑规避限制。论文300步与脚本301、脚本4卡与附录2卡、paper状态进度奖励与released动作前缀奖励的差异已在报告32明示，第一条线保持CODE_AS_RELEASED。该训练尚未启动，本次不能当作其性能或预算可行性的证据。
