# 带检索技能输入的原生 actor/critic PPO 更新：限定核查通过

本次用户要求的补充更新核查已完成。128条固定训练记录中51条带检索技能，actor与critic均在真实native PPO更新入口通过响应token、mask、检索ID及processed图像grid/pixel结构核对；两个role均完成4次原生分布式优化器更新，损失/梯度有限，固定本地参数样本有非零变化。只完成global_step1→2这一批，未跑301步。

## 真实更新与证据

| 检查 | actor | critic |
|---|---:|---:|
| 实际输入 / 带检索技能 | 128 / 51 | 128 / 51 |
| 每rank原生optimizer调用 | 4 | 4 |
| 原生平均损失 | -0.000262633 | 0.038098313 |
| 平均梯度范数 | 6.304222 | 74.072320 |
| 非零变化 / 固定参数样本 | 35148 / 35328 | 35313 / 35328 |
| 原LR | 1e-06 | 1e-05 |


两个rank承载同一个分布式更新过程，不能把每rank4次计成8次独立更新。固定样本为每个非空本地trainable参数分片的前256元素，并非全模型逐元素审计；保留每个rank完整统计与未变化元素。

完整actor/critic model、optimizer、RNG/scheduler和dataloader从原step1恢复；实际下一批种子与调用前CPU冻结的128逐项一致，与首批不重叠。回复token与训练mask逐行对应，padding mask为0，old logprob/advantage/return在参与损失的位置有限；51条检索输入确实进入两个role的原生更新。输入图像网格乘积与pixel行数一致，带检索行同时包含环境和技能图像。

## 原失败和恢复没有被覆盖

首轮因我新增的actor输入证据torch.save代码不能保存共享存储的不同dtype视图，在actor native update前退出；critic已更新但未保存活跃状态。原失败、128局/415条回复与全部成本见[报告39](39_retrieved_input_audit_incident_and_replay_protocol.md)，不是原生PPO算法已被证实有错误。

修复只在保存副本时逐tensor clone，未修改传给原生更新的数据或算法。恢复使用全部原128局已记录的回复，不按结果换种子、不重新生成；CPU重建匹配全部415个prompt token IDs、采样参数、图像数量及最终回复/mask/检索/环境状态/动作/位置/distill和奖励。两次CPU装配失败也保留说明。原生AgentLoop后处理生成私有重建输入bank，然后从相同step1完整状态重新计算old logprob、values、GAE并完成critic与actor更新。

恢复新增模型生成请求0；首轮critic重做是恢复成本，不是独立重复证据。重建并未逐字节对比丢失的原processed pixel tensors，因此不把它称为原actor输入bank完整无损恢复。两份本次真实actor有效输入bank保留服务器，保存副本clone后的值不参与训练计算。

## 时间、成本和资源

- 实际于16:51:04完成关闭，退出码0、reason为空、完成阶段仅train，4份role/rank审计均完成；完整全状态恢复与global_step1→2回执齐全，无本轮残留活动进程，argv违规0。
- 恢复supervisor总历时1183.29秒（19分43秒）；从原补充准备时点至退出5186.91秒（1小时26分27秒），未重置原17:24:38硬截止。原父24小时截止也保留。
- 本次native更新路径220.59秒，其中critic72.20秒、actor72.28秒；完整状态重载约15分51秒。原metrics中的gen27.43秒是已记录bank读取/后端wake-sleep，不代表模型生成吞吐。
- 新增模型生成请求0。首轮实际415次生成请求全部保留；加上此前父轮/空库首批核查，合计4239次实际生成dispatch、1152次原始采集episode执行，仅作成本口径；另128次CPU环境重放。critic首轮完成后在恢复中重新执行的计算另计，不冒充更多独立样本。
- 关闭后含小型证据导出四根155,271,806,976 bytes，较补充原基线新增1,896,701,952 bytes（约1.90GB），低于共享严格8GB；相对父基线新增110,869,012,480 bytes，低于用户本次150GB例外/总量210GB。旧模型与历史结果未删。
- 真实actor输入bank两个rank分别411,062,715和412,455,163 bytes，CPU重建bank1,050,185,307 bytes，均保留服务器；未重新保存整套模型/optimizer。

原step1约108.90GB完整恢复点保留只读；本次step2终止诊断没有保存另一套模型/optimizer，因此没有可续训step2模型。公开提供源代码、协议、审计、metrics、128局结果及完整原415条回复/output IDs/采样记录；private effective-input bank、模型/optimizer、批量prompt IDs和完整运行日志留在受控服务器。

## 最终结论边界

在本次限定范围内，没有发现检索技能未进入更新、响应与mask错位、梯度/损失非有限、优化器未执行或参数未变化的问题。已发现的我方审计保存错误已修复，补充核查通过。

这不证明所有潜在实现错误都已排除，也不验证301步性能、长期稳定性、采样端与训练端后端的全面数值等价。此前报告18/19负证据及greedy重启差异仍有效。成功率不作为本次通过门槛，采集32/128成功属于更新前行为，不是本次更新后的性能结果。

用户指定的核查范围至此完成，不自动启动301步或新方法；完成后移除本会话对应监测。

[完整结果JSON](../docs/evidence/reproduction/retrieved_input_ppo_replay_v1_result/result.json) · [故障与调用前恢复协议](39_retrieved_input_audit_incident_and_replay_protocol.md)
