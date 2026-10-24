# 保存请求核对完成：全token概率门失败，首批信号出现可检验风险

8条首轮/第二轮跨后端概率及原B首批32条训练信号已完成。**8/8未达到调用前冻结的全token误差门；差异主要出现在think，所测动作字段非常接近。首批两个Stop动作token在原critic下获得正白化优势，在V=0反事实下转负。** 这些结果支持有限机制诊断，不证明原PPO退化原因已经找到，也不改变报告16的学习失败。

本次没有新的环境动作、训练采样、optimizer更新或入库。发生的接口/适配器/CPU分析故障全部保留；原生实际prefill8个、RPC尝试9次，初次接口意外返回1个未使用换行token，原零生成及8-dispatch契约均不判PASS。

## 1. 冻结样本、生产路径与输入边界

B theta0、有卡上下文：按seed升序取两个至少有两轮记录的条件三步任务110002、110074，以及两个自然多步无检索卡任务61001、61002；各第1、2轮，共8条。没有按成功、Stop或误差选样。

原生评分使用SGLang的模型、处理器、hybrid同步b0和原生input-token logprob。actor使用实际FSDP、fused kernel、remove_padding及原训练拼接：4条完整轨迹，另4条截至第二轮的prefix控制，共8个actor样本。8条原生因果前缀分别与完整训练布局对应位置比较；第二轮另报告prefix actor控制。没有用standalone HF冒充production actor。

中间观测图片没有历史落盘，按保存状态/动作朝向用原确定性渲染器重建，不调用env.step。36条轨迹共72次保存首末PNG像素核对通过；保存token、完整response与generated/observation mask逐项一致。**历史中间图像张量身份仍未完全证明。** 全部实际使用图片及路径映射、精确token和数值输出在公开小型证据包中。

## 2. 跨后端结果

调用前操作门：max绝对误差≤.05 nats，mean≤.005。保持原门，不根据结果放宽。

| seed | 分组 | 轮次 | max绝对误差 | mean绝对误差 | 原门 |
|---|---|---:|---:|---:|---|
| 110002 | conditional | 1 | 0.152977 | 0.024151 | FAIL |
| 110002 | conditional | 2 | 0.112337 | 0.018198 | FAIL |
| 110074 | conditional | 1 | 0.080110 | 0.013151 | FAIL |
| 110074 | conditional | 2 | 0.228028 | 0.021598 | FAIL |
| 61001 | natural | 1 | 0.127096 | 0.020090 | FAIL |
| 61001 | natural | 2 | 0.079106 | 0.011348 | FAIL |
| 61002 | natural | 1 | 0.084546 | 0.018876 | FAIL |
| 61002 | natural | 2 | 0.103314 | 0.016086 | FAIL |

全部8条prompt token计数符合原始输入长度，响应token IDs逐项对齐且有限。第二轮完整布局与prefix actor之间的max差为0.109375、0.09375、0、0.103515625 nats；改用prefix仍不能使对应跨后端全token误差通过。

按411个原响应token分层：

| 部分 | token数 | mean绝对误差 | max绝对误差 | actor/native比率超出[.8,1.2] |
|---|---:|---:|---:|---:|
| think |311|.023280|.228028|1|
| 动作字段 |8|.000390|.001381|0|
| EOS |8|.00000285|.00001037|0|
| 格式/角色 |84|.001597|.049162|0|

因此，不支持“已观测到正确动作与Stop之间巨大概率失配”这一说法。这8条动作位置没有Stop，无法替代对Stop分支的专门测量；think本身参与PPO损失，误差也不能被忽略。一个think token的比率低至.7961，但旧PPO的old logprob由actor重算，不能据此说它直接触发了旧运行的clipping。也不能由8个b0请求证明已丢失theta4或所有长历史都正确。

## 3. 原首批32条信号重建

原b0骨干＋保存的完整小critic head2049个值恢复。6917个生成token；原始成功2/32，两个成功均为单步，目标三步0/8。

重建的七项聚合指标与原第一批日志逐项相同：BF16汇总value mean3.625、max9.625、min−3.484375，return mean.0839526728，白化优势mean/min/max。当前无损转FP32计算的value mean3.630133867，std2.234916210。该吻合支持重建可信度，仍不等于全部历史张量身份被逐项验证。

需要纠正历史解释：报告17引用的critic/vpred_mean≈2.3574来自critic更新各micro/minibatch的预测聚合，**不是更新前compute_values的初始均值**。更新前应看critic/values/mean。不能把两个指标混为一谈。

FP32生产奖励路径下，return恒等误差5.327e−7；给values统一减去其均值后，白化优势最大差2.384e−7。统一均值偏移确实被白化抵消，不能靠“均值偏高”归因。

首批仅有两个纯Stop回答，各恰好1个动作token：

| Stop所在类别 | 即时奖励 | 原白化优势 | V=0反事实 | 按回合未来回报反事实 |
|---|---:|---:|---:|---:|
| 目标三步 |0|+1.540409|−.507098|+1.542678|
| BFS≥4 |0|+1.690106|−.189927|+1.674440|

这里是状态/token相关value差异的具体信号；按回合分配奖励没有消除这两个正信号，V=0改变了符号。但只有两个Stop，且共享参数会受到整批其他token的梯度影响。**优势正负不能证明一次Adam更新后的Stop概率方向；仍需单一干预的一次更新机制对照，不能宣布critic已经被证实为根因。** microbatch1的actor损失先在每episode内token平均再累积；不能把长轨迹token数直接当整批线性权重。

## 4. 保留的故障与成本

1. 初次初始化：Ray显式runtime_env把项目路径序列化入子进程argv，被中性命令行守卫终止。0评分前向。修复为启动Ray前普通环境继承，独立0GPU检查通过。
2. 首次原生max_new_tokens0：安装版仍返回换行ID198、completion_tokens1、finish length0。硬失败保留，原输入概率已计算，第1条未重算。
3. 评分适配器：通过原生TP worker已有skip_sample=True和input-logprob处理，在同步无radix cache的prefill中真正跳过采样；不改安装包或PPO。一次字段兼容错误发生于实际模型前向前；仅此请求重试一次，保留9次RPC/8次真实prefill和原8-dispatch契约失败。剩余7条实际返回0token。普通生成仍委托原实现，CPU检查包含该接口边界。
4. 首批前向完成后的CPU分析：我的重建脚本误用BF16奖励，产生return误差.158203125及shift误差.015625。**这是新审计脚本错误，不是原PPO错误。** 原生产奖励为FP32；使用已保存32条actor/critic输出在CPU纠正，无额外模型前向、不改门槛，原失败与数值保留。

实际选定样本计算：SGLang8个prefill、actor评分8个样本、首批actor32＋critic32个前向；环境动作0、optimizer0、入库0，意外新token1。所有初始化、失败、追加RPC和CPU恢复均计入本次首次准备时间。原driver因上述硬检查/分析错误退出1，最终COMPLETE由已保存张量的CPU恢复写入，不冒充driver全程exit0。

## 5. 资源与后继决定

首次准备2026-10-24 23:21:35不重置。用户随后明确授权12小时自主实验、不受旧GPU窗口限制，故在评分前单列资源修订：保留旧30分钟与旧campaign截止为历史，本组件最多2小时含全部准备，截止2026-10-25 01:21:35；500MB预算/450MB写停和科学判据不改。新总窗口为2026-10-24 23:38:27至2026-10-25 11:38:27，新增累计严格<50GB，另守项目85GB保守线与GPU3/4/5边界，唯一半小时跟进。

下一步不扩展旧4批训练、不自动300步。先区分共享因果前缀的输入身份与kernel/精度差异；若后继仍做训练，只在明确证据支持一个干预、且冻结的新32条数据确有目标成功信号时，开展同起点/同数据的一次更新机制对照。零value-head初始化是有依据的候选，按回合奖励暂不优先；不同时改lr、clip、KL或奖励，不把恢复到b0或仅好于旧崩落B称为学习成功。

证据：[汇总](../docs/evidence/components/recorded_backend_audit_v2/summary.json)、[8条逐token概率](../docs/evidence/components/recorded_backend_audit_v2/score_results.json)、[逐回合优势](../docs/evidence/components/recorded_backend_audit_v2/training_signal_audit.json)、[完整数值信号NPZ](../docs/evidence/components/recorded_backend_audit_v2/first_batch_signals.npz)、[历史聚合核对](../docs/evidence/components/recorded_backend_audit_v2/historical_aggregate_check.json)、[事故摘要](../docs/evidence/components/recorded_backend_audit_v2/incidents/summary.json)。NPZ是模型输出/训练信号，不含模型权重。
