# 带检索技能输入的原生 PPO：单批正确性核查协议

用户要求继续排除关键实现错误，不需要跑满301步。本次只从报告37已验证的global_step_1完整恢复点继续下一批128个训练样本；不启动新方法，不按成功样本补采，也不新增验证集查询。以下协议与CPU检查先于本次GPU调用冻结，执行源码 `5cdeb6d49611445a646c4d2db2f2dbbb6d2dea5f`。

## 原生路径和检查条件

- 使用已保存actor/critic的model、optimizer、extra中的RNG/scheduler，以及原生StatefulDataLoader；10条已有在线技能复制到本次库，原完整恢复点只读保留。
- 下一批128种子从真实data.pt恢复后在CPU冻结，与原首批128无交集；实际控制器必须逐项匹配才生成。冷启动后端的逐token确定性不作保证。
- 保留原GAE、full generated response mask、FP32 AdamW、batch128/mini32、actor1e-6/critic1e-5、温度1、skill_guided .8和原self-distill。总目标301保留用于原scheduler语义，实际只运行global_step2这一批。
- 在真正native actor/critic update入口逐行对照已记录episode的response IDs、mask和retrieved_skill_id，检查padding mask、参与损失位置的old_log_probs/advantages/returns有限，以及processed pixel/grid形状。检索行必须包含环境与技能图像。
- 原生优化器使用无修改的post-hook计数：每role每rank恰好4次step，两个rank是同一次分布式更新的分片，不能计成8个独立step。记录有限原生loss/grad与每个非空本地参数分片预先固定前256元素的非零变化；不声称检查全部参数。
- 两个role各128条完整输入均需通过，且确有检索技能输入行进入两者的实际更新。若没有检索覆盖或检查失败则关闭并保留证据，不额外采样或调参补通过。

## 预算、终态和边界

新增最多128局、640个rollout请求与128个self-distill请求，共768；没有验证pass。新增严格小于8GB，6GB停止写入；此前父轮150GB新增/210GB四根总量例外及原始24小时截止继续约束所有成本。本次额外窗口从15:24:38开始，17:09:38协作停止、17:24:38硬截止，准备/修复计时不重置。

只保存实际actor输入张量到受控服务器（每rank张量预算小于2.5GB）、数值审计和完整请求/episode证据；不再复制约109GB完整状态。global_step_1的完整恢复点始终保留，step2结束活跃权重是终止用诊断状态，**没有保存step2可续训模型或optimizer**。不将它描述为最终性能模型。

CPU检查已实际通过，包括真实下一批sampler恢复、MRO/config、2x2线性参数的4次CPU AdamW hook计数、原返回对象保留，以及错误token拒绝。CPU fixture没有实际模型forward/backward、GPU调用或环境采样；真实GPU结论仍待运行。

既有报告18/19后端数值负证据和报告37的greedy重启差异仍有效。本核查不证明所有采样/训练输入的数值等价、长期稳定性、论文性能或所有潜在错误均已排除。

[冻结协议](../configs/reproduction/retrieved_input_ppo_check_v1.json) · [CPU回执](../docs/evidence/reproduction/retrieved_input_ppo_check_v1_prereg/cpu_check.json) · [下一批种子](../docs/evidence/reproduction/retrieved_input_ppo_check_v1_prereg/next_batch_manifest.json)
