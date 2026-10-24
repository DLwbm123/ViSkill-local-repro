# 关闭融合输出内核未消除跨后端概率差异

冻结的8个actor样本前向已完整结束，exit0；**8/8仍未通过既有max≤.05、mean≤.005 nats门，误差与原融合内核接近。该有限对照不支持融合输出内核是主要差异来源。** 不继续搜索精度或内核组合，不改变报告18的原失败，也未进行训练。

## 输入与唯一干预

沿用报告18相同b0、四条完整轨迹＋四条截至第二轮prefix，共8个actor样本；对比原先已保存的8条原生SGLang概率，没有新原生请求。当前重建输入的四组共享因果前缀token、有效attention位置编码、image_grid_thw及pixel_values逐项相等，CPU门通过。该检查不能补回缺失的历史中间张量。

唯一数值干预为model.use_fused_kernels=False及actor消费项False，仍使用BF16/FSDP/FA2/remove_padding和实际compute_log_prob。只初始化actor，不建rollout或critic；配置GRPO仅为跳过critic，不计算GRPO或任何更新。官方FrozenLake base/skill脚本本身均显式启用融合内核，因此这不是修复本地擅加配置，也不证明官方模型存在缺陷。

## 结果

| seed | 轮 | 融合max | 参考max | 融合mean | 参考mean | 原门 |
|---|---:|---:|---:|---:|---:|---|
|110002|1|0.152977|0.156280|0.024151|0.024015|FAIL|
|110002|2|0.112337|0.111942|0.018198|0.018124|FAIL|
|110074|1|0.080110|0.080442|0.013151|0.013202|FAIL|
|110074|2|0.228028|0.226295|0.021598|0.021735|FAIL|
|61001|1|0.127096|0.124069|0.020090|0.019986|FAIL|
|61001|2|0.079106|0.075293|0.011348|0.011403|FAIL|
|61002|1|0.084546|0.084248|0.018876|0.018671|FAIL|
|61002|2|0.103314|0.104176|0.016086|0.016199|FAIL|

第二轮完整布局与prefix的max差仍为0.113232、0.096512、0.000000、0.105095 nats。共享输入前缀相同不意味着不同长度BF16计算输出逐位一致；本次未隔离其余注意力/累积数值来源，也不据此宣布根因。

## 成本、边界与后继

首次准备2026-10-25 00:38:32，完成00:41:35，清理退出00:41:46；未重置时钟。8个actor前向样本，0新原生请求、0生成token、0环境动作、0optimizer更新、0checkpoint。运行守卫记录的新增峰值含准备2,392,064 bytes；公开数值输出整理后观察新增约2.41MB，仍低于500MB预算/450MB写停，独立含准备上限1小时；沿用11:38:27总截止与累计新增<50GB/项目85GB保守线。命令行守卫无违规。

决定关闭这一kernel参考支线。报告18的两个Stop优势在V=0下转负仍只是可检验风险，下一优先候选是同起点、同新采样数据、同优化器初态、仅改变初始value head的一次global actor update对照。该后继必须另冻结，先获得独立新32条(16目标三步＋8单步＋8其他BFS2/3)且有目标成功；若没有目标成功，不更新、不重采样补阳性。不得由只好于旧崩落B或恢复到b0宣称学习有效，不自动扩大常规训练。

证据：[逐token结果](../docs/evidence/components/kernel_score_reference_v1/results.json)、[输入前缀检查](../docs/evidence/components/kernel_score_reference_v1/prefix_identity.json)、[数值输出NPZ](../docs/evidence/components/kernel_score_reference_v1/actor_outputs.npz)、[退出与成本](../docs/evidence/components/kernel_score_reference_v1/closure.json)。NPZ仅含模型输出，不含权重。
