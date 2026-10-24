# 新地图前瞻性验证失败：零值头未超过原始b0，关闭本性能线索

**在另行冻结的32张新地图、三政策共同逐局串行评估中，条件任务成功/质量：b0为7/6（各/16）、原值头O为3/3、零值头Z为4/4。Z低于b0，其他自然多步的Stop回合4→10，预设前瞻性投入门失败。报告20的单例改善未获得本次支持；关闭零值头单步性能线索，不补seed、不追加本协议训练或再次挑图。**

完整96局、364次原生调用；新增训练、蒸馏、入库和模型快照均0。O/Z保留模型已实际成功加载并用于新图推理；这不证明完整FP32/优化器恢复，也不改写报告18的跨后端概率失败。

## 1. 前瞻性设计与输入控制

在任何本轮业务模型调用前冻结：按200001..216384候选seed升序首次选16个Up3适用/BFS3/top1 skill4得分≥.8的条件地图，另16个不满足该Up3适用条件的其他自然多步地图。以整张desc地图身份排除12,407个历史地图、13,766个实际历史候选前缀seed及全部保留Test；32张唯一、无历史重合。全筛选与初始状态见[冻结数据](../docs/evidence/components/prospective_value_head_eval_v1/data_manifest.json)。不据模型输出挑图，没有原保留Test的模型查询。

b0、O、Z使用相同新panel/顺序、固定5卡、生产hybrid agent/prompt/processor/SGLang、T0/top_p1、fused/BF16。val_batch=1、agentworkers=1且实际loader32×1，每次只1个环境episode活动；串行调度为控制此前greedy批调度变异而预先固定，共用于三政策。**本panel与报告20不同且调度也不同，不能把两个panel的绝对成功率变化归因为串行或学习。** 新panel内配对比较才是本报告的主证据。

O/Z均来自报告20同批数据、同b0、各一次actor更新的已保留模型，没有新优化。实际HF初始化捕获loading_info：三政策missing/unexpected/mismatched/error全空，且输入/输出embedding为实际同一Parameter；不存在失败后回退b0。CPU检查三者chat_template、special tokens、generation metadata及canonical双图的input_ids/attention/pixel_values/image_grid逐项相同。快照导出遗漏的216-byte generation_config在首次新推理前从原b0补齐，权重未改；统一原EOS/PAD和默认生成元数据。只验证了实际加载可用性，没有全权重hash或字节身份声明。

## 2. 全部结果与配对退步

| 分组 | 政策 | 成功 | BFS路径质量 | 含Stop局 | Stop回合 | hole局 |
|---|---|---:|---:|---:|---:|---:|
|条件16|b0|7|6|4|16|4|
|条件16|O|3|3|8|29|3|
|条件16|Z|4|4|6|24|2|
|其他自然16|b0|2|2|1|4|5|
|其他自然16|O|2|2|2|8|5|
|其他自然16|Z|3|3|3|10|5|
|全部32|b0|9|8|5|20|9|
|全部32|O|5|5|10|37|8|
|全部32|Z|7|7|9|34|7|

条件16张上，Z相对b0成功3改善/6退步，路径质量3改善/5退步；相对O成功与质量均3改善/2退步。Z虽然比O多1个条件成功，仍低于原始b0，且没有达到“较两者各至少+2、退步各≤1”的门。不能用好于另一个已退化分支掩盖低于b0。

Z相对b0的条件质量差−12.5个百分点，保守有限样本95%配对差区间[−58.89,+39.95]个百分点；条件成功差−18.75个百分点，区间[−64.61,+36.12]。全部32张质量差−3.125个百分点，区间[−32.41,+27.00]。区间宽且跨0，不能声称已统计证明普遍伤害；它们同样不支持稳定提升，预先定义的投入门明确未达到。

其他自然多步质量2→3、1改善/0退步，但Stop回合4→10、含Stop局1→3，违反自然安全门；hole5→5未变。总成功Z7/32低于b0的9/32；总路径质量Z7/32低于b0的8/32。完整每seed记录与全部CI/sign p见[结果](../docs/evidence/components/prospective_value_head_eval_v1/result.json)。区间方法沿用报告20的CP/Bonferroni保守配对差界，未据本轮结果放宽。

## 3. 冻结门与结论边界

已达到：Z条件绝对质量4/16；自然成功/质量计数不降且原有成功/合格局0退步；自然hole不增加。

未达到：相对b0与O各至少+2条件质量、相对两者的条件成功/质量退步≤1、自然Stop回合/含Stop局不增。因此最终gate=false，关闭本零值头性能线索，不追加新训练/seed或把第20报告的1个改善当确认结果。历史Stop迁移机制门仍FAIL，新性能结果不能救回原机制解释。

这不是原论文300-step实验复现，也不是医学任务结论。仍只有一个共享训练批/每政策一次更新，无训练seed复制；既有native/actor think概率门失败仍保留。下一步仅先对保存的训练信号作CPU损失支持审计，判断是否存在有依据的独立主线机制假设；不为用满授权窗口继续GPU，不把新假设当已经验证。

## 4. 保留的零业务请求故障与实际成本

原b0模型初始化及SGlang warmup后，新EvalTrainer直接借用另一trainer定义的_get_gen_batch，零参数super保留原定义类闭包，首次gen-batch转换抛TypeError。调用尚未进入manager.generate：0业务native请求、0环境episode、0优化步骤；**初始化/warmup确实消耗GPU计算**，不声称0GPU成本。原退出1与全部文件保留。

CPU用真实首条DataProto复现旧super错误并验证新own-method后，仅一次before-generation repair；原始起点/数据/门/预算不重置，未重播任何已完成case/API。三个修复阶段最终均exit0、无argv违规；原“没有失败尝试”不判PASS。另一次CPU错误import也在0模型业务调用下修正并保留。加载检查不是替代性能门。

| 实际政策 | 环境局 | 原生调用 |
|---|---:|---:|
| b0 |32|117|
| O |32|125|
| Z |32|122|
|合计|96|364|

首次准备01:52:47，GPU最终02:29:55退出，原含准备2小时截止03:52:47不重置；所有阶段共享96局/480调用和500MB硬预算/450MB写停。运行守卫新增峰值17,829,888 bytes，公开证据整理后新增25,874,432 bytes，总占用约83.98GB，低于组件84.40GB写停与项目85GB线。关闭时本组件残留进程0。12小时总截止11:38:27、累计新增严格<50GB保持；O/Z两模型及旧b0/r0/32.56GB恢复checkpoint均未删除。

证据：[完整结果/门/配对区间](../docs/evidence/components/prospective_value_head_eval_v1/result.json)、[CPU输入控制](../docs/evidence/components/prospective_value_head_eval_v1/cpu_runner_check.json)、[实际Z模型加载](../docs/evidence/components/prospective_value_head_eval_v1/Z/actual_loading_info.json)、[初始零请求失败](../docs/evidence/components/prospective_value_head_eval_v1/before_generation_failure_audit.json)、[退出/存储](../docs/evidence/components/prospective_value_head_eval_v1/closure.json)。公开全部96局/364调用的合成游戏证据与源/config；不公开模型权重、私有头、运行token或完整runtime日志。
