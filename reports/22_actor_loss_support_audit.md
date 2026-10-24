# 保存批次的损失支持审计：非答案区域占94.9%与96.5%的标量系数

事后CPU审计覆盖报告20同一32条训练数据，6329个实际生成token，其中233个与答案内容区间重叠。O、Z分支的非答案区域分别占绝对logprob标量系数质量94.89%和96.48%；全部32条都有非空答案候选mask。**这支持另立一个“直接actor损失区域”单因素假设的可执行性，不是参数梯度范数、Adam影响或Stop退化因果证明。** 零值头前瞻性性能门和历史Stop机制门依然失败。

## 1. 实际代码分支与系数

已读取两分支真实resolved配置：global mini32、micro1、token-mean、ppo_epochs1、非dynamic batch、vanilla、entropy=0、KL关闭、use_rollout_log_probs=False。worker按2-rank将mini32归一为16；每rank只一个minibatch，因此安装实现的on_policy快捷分支把old_log_prob设为当前log_prob.detach，ratio恒为1。该变量名不是原生SGLang与actor严格on-policy等价证明。

所以本次单步、未跨minibatch更新时，对生成位置的标量系数为：

`dL/dlogp[i,t] = -advantages[i,t] / (32 * N_generated[i])`

这是micro1内按生成token均值、再global32平均的结果。利用实际vanilla loss做了两个小型CPU标量autograd回归（正/负优势与mask0位置），符合该公式；没有模型初始化、模型前向/反向或优化步骤。

由此可知报告20的clipfrac=0/ppo_kl=0在此分支下是预期的，不能据它们确认跨后端概率对齐或参数未变。旧4批mini8协议每rank有多个minibatch，不直接套用此结论。源码见[actor实现](../verl/verl/workers/actor/dp_actor.py)。

## 2. 完整角色分层

角色由原token IDs、generated spans、完整response_mask和逐前缀解码的字符边界确定；观察/padding完全排除。答案重叠可能包含标点与标签边界，不称为纯动作logit。所有生成位置恰好分配一次。

| 区域 | token数 | O绝对系数质量占比 | Z绝对系数质量占比 |
|---|---:|---:|---:|
|答案内容重叠|233|5.11%|3.52%|
|思考内容重叠|4993|60.09%|76.08%|
|EOS|96|1.06%|1.77%|
|格式/角色/未解析|1007|33.74%|18.63%|

输入token与保存response/mask逐项匹配，32条都可构造非空答案mask，没有丢弃、替换或重新采样。该审计的“可行性门”只要求覆盖全部32及两分支非答案质量≥90%，结果达到；它不评价训练效果。所有每episode计数、未解析回合与符号/绝对系数在[完整结果](../docs/evidence/components/actor_loss_support_audit_v1/result.json)，候选mask/系数在[数值NPZ](../docs/evidence/components/actor_loss_support_audit_v1/loss_support.npz)。这里都是既有数据与损失的派生输出，不是模型权重。

## 3. 不支持的推断与后继边界

标量系数还要乘每个token对模型参数的Jacobian，并经共享参数累积、梯度裁剪、Adam处理；94.9%/96.5%不等于相同比例的参数梯度或训练破坏。不能据此说“思考导致Stop”或“答案区域损失必然更好”。新训练批本就没有Stop，历史探针不是同Stop样本直接梯度实验。

下一候选仅是独立机制试验设计：保持原全生成mask计算GAE与whitening，固定同b0/C32/优化器初态/顺序/LR/clip，唯一改动为actor直接损失使用答案内容mask；不同时改奖励/模型/学习率/Stop集合，不筛样本、不扩常规训练。必须先冻结新前瞻性地图与绝对/相对/自然安全门，并保证新模型可恢复且满足85GB存储线。目前尚未冻结或启动该GPU试验，不追加已关闭零值头协议训练。

原有O/Z快照及b0/r0/32.56GB恢复checkpoint继续保护。可先在CPU上对一个固定已保存模型验证精确BF16差分编码的存储可行性，作为恢复工程准备；编码中的学习参数仍属于私有权重，不公开发布，不用压缩可行性替代性能证据。

本审计首次准备03:22:49，15分钟含准备上限保持，CPU结果03:25:45完成；0模型forward、0环境动作、0模型backward、0optimizer，只有两个小标量autograd fixture。后验描述与原失败均保留，不声称独立验证或原论文复现。
