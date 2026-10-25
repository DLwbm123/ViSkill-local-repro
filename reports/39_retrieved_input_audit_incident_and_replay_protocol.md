# 检索输入核查中断：我方输入证据保存错误与固定记录恢复协议

**首轮带检索输入核查未通过整体完成条件。** 128局采集全部完成，其中51局确有检索技能；critic两rank各完成4次原生optimizer.step、有限梯度与非零参数样本变化。随后我方新增actor输入证据保存代码触发 `Cannot save multiple tensors or storages that view the same data as different types`，发生在native actor更新前；actor无完成审计，不能判整体PASS。

## 已完成与原故障

原生完整actor+critic model/optimizer/RNG/scheduler/dataloader恢复返回，global_step1。critic128条输入的响应token/mask、检索ID与processed pixel/grid形状核对通过，其中51条带检索。两个rank分别17664个参数样本，共35328，35317发生非零变化；4个分布式梯度范数为58.6162、97.4063、32.2049、108.2686，有限且非零。两个rank同属4次分布式optimizer更新，不能合计为8次独立更新。

本轮415次实际模型请求=406rollout+9self-distill，0请求错误，24976输出token；128局采集成功32（更新前行为，只作描述，不是正确性门槛）。supervisor于16:03:22退出1，实际运行13分51秒。无本轮残留活动进程、argv违规0。critic更新后活跃状态未保存；原step1完整恢复点保留。792字节的失败输入文件不能冒充有效输入bank。

根因属于我新增的审计代码：输入batch含共享底层存储且dtype不同的tensor视图，直接torch.save容器被拒绝。修复只对保存副本逐tensor独立clone，不修改交给原生更新的data对象、PPO计算、优化器或超参。CPU最小fixture实际复现原错误，修复后dtype和值roundtrip通过。原失败执行源码5cdeb6d49611445a646c4d2db2f2dbbb6d2dea5f与全部请求/episode、critic审计和成本保留，未追改为成功。

## 恢复前已通过的CPU证据

不重新调用模型、不换128种子，使用同一库快照10条技能、同一环境和processor，逐条重放已有415个输出token序列。128局均匹配全部原prompt token IDs、采样参数、图像数量；最终response IDs、mask、检索ID、初始状态、动作、位置、distill文本与奖励也匹配。重建经过原生AgentLoop后处理，生成约1.050GB私有输入bank，保留服务器。

两次CPU装配尝试分别在任何episode开始前被重复agent_name参数和Ray装饰类绑定方式错误中止，已修复；之后全部128局CPU环境重放完成，实际模型forward/backward、GPU调用和新模型生成请求均0。重建不等于丢失的原processed pixel tensor已逐字节比对；这个限制保留。

## 调用前冻结的唯一恢复

从相同原生完整step1状态重载一次，传入上述CPU验证的全部128条记录输出，继续同一native训练路径：重新计算old logprob、values、GAE，完成原critic更新与尚待核查的actor更新；保留batch128/mini32、FP32 AdamW、原LR/原301-step scheduler语义，仅停在global_step2。原native更新计算不改写。

生成入口只能消费已经保存的固定batch一次，允许原后端wake/sleep管理但绝不调用generate server；新模型生成请求上限0，遇错无采样fallback。不把重复执行的critic更新视作独立结果，也不把CPU重放episode加到独立数据样本数。两个role四份rank审计均需128完整行、51检索覆盖、各4次优化器调用、有限损失/梯度与非零固定参数样本变化。step2仍不保存另一套完整模型/optimizer，step1恢复点只读保留。

原额外2小时窗口15:24:38→17:24:38不重置，17:09:38协作停止；原父24小时截止继续绑定。全部首轮失败、CPU重建与恢复共享新增严格8GB/6GB写停止线，以及父150GB新增/210GB四根例外。目前含小型失败导出与私有重建bank四根154,438,930,432 bytes，较补充基线新增1,063,825,408 bytes，保留余量。

恢复完成前没有actor带检索PPO通过结论。旧后端数值负结果、greedy重启差异、所有参数或长期稳定性边界均未消失。

[首轮完整结果](../docs/evidence/reproduction/retrieved_input_ppo_check_v1_result/result.json) · [冻结恢复协议](../configs/reproduction/retrieved_input_ppo_replay_v1.json) · [CPU重建回执](../docs/evidence/reproduction/retrieved_input_ppo_replay_v1_prereg/cpu_check.json)

恢复执行源码 `742d38fc6d727bfe8b93ba9cb9806951f62a0c94`；本协议先于恢复GPU调用发布。

实际恢复于16:51:04退出0，actor与critic四份审计均通过，新增生成请求0；完整结果及保留边界见[报告40](40_retrieved_input_native_ppo_correctness_result.md)。原首轮故障不覆盖。
