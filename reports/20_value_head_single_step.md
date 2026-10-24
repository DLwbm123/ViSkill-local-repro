# 零值头单步仅新增1个成功；历史Stop迁移机制门失败

**同一新32条采样、同b0的一次actor更新，零值头Z在8张条件Dev上成功/质量0→1，原值头O为0→0；其他自然多步均1/8持平。预设有限后继评估门达到，但只是1个地图级改进，配对区间跨0，未建立稳定学习收益。两个历史Stop实际概率对照均未按预期改善，机制门失败。** 不改变旧固定库PPO失败，也不追加旧训练批次。

本轮完整完成96局、338次原生调用；两分支各一次global actor step，无critic更新、无入库。一次新审计代码的输出序列化错误在更新前发生，保留全部原失败与成本，未重复完成的环境采样或actor更新。

## 1. 冻结的数据与干预

按seed升序首次选择16个新Up3适用三步目标、8个单步、8个其他BFS2/3，共32条；另冻结8个条件Dev和8个非Up3适用的其他自然多步Dev。以整张desc地图身份排除10,369个历史地图、11,298个已实际筛选候选seed及全部保留Test；新48张地图唯一、无train/Dev/历史重合。没有按模型输出挑图。

C实际采样成功13/32：目标7/16、单步4/8、其他BFS2/3为2/8，达到“至少1目标成功才更新”的门。32条全保留、不重采样填阳性。保存了实际生产input IDs、位置、mask和处理后的多模态张量，见[精确输入NPZ](../docs/evidence/components/value_head_single_step_v1/C/actual_inputs.npz)；本批无需按历史状态重建中间图像。

O与Z共用这一份实际native-b0行为数据，actor从同一b0、空优化器、相同顺序开始；唯一干预是保存的原始2049参数critic小头或全零小头。实际初始actor整批logprob逐项相同，max/mean差均0。使用原LR=1e-6、clip、GAE1/1、终局合计奖励、固定5张图卡、0技能奖励/0 KL；FSDP/fused/remove_padding/BF16不变。每分支仅一次global mini32/micro1 actor更新，与旧mini8协议明确不同。critic只前向、不更新，所以这是单步机制试验，不是完整常规PPO。

## 2. 完整Dev结果

成功与BFS路径质量计数本轮完全相同。

| 分支 | 条件8张 前→后 | 条件改进/退步 | 其他自然多步8张 前→后 | 全16张 前→后 | 全16张Stop回合 前→后 | 全16张hole 前→后 |
|---|---:|---:|---:|---:|---:|---:|
| 原值头O |0→0|0/0|1→1|1→1|15→15|5→5|
| 零值头Z |0→1|1/0|1→1|1→2|19→11|5→4|

Z新增成功为seed150280，执行3步；原来唯一的自然成功150002保留，没有成功退步。Z在其他自然多步上的Stop回合3→3、含Stop局数1→1、hole2→1；达到已冻结的自然安全计数门。

Z条件质量净变化+12.5个百分点，保守有限样本95%配对差区间为[−42.02,+57.55]个百分点，精确discordant符号检验p=1；全16张净+6.25个百分点，区间[−23.88,+33.79]。这些区间很宽，不能声称显著、稳定或普遍提升。区间由win/loss两个97.5% Clopper-Pearson界的Bonferroni组合构造，允许配对内依赖；计算细节在初始采样及原O基线后、查看post结果前明确，未据此修改原性能门。

还必须保留一个运行限制：O、Z即使同b0，greedy基线的Stop回合也不同(15与19)，因此不能把跨实例基线当完全一致。原生/actor的think概率误差仍是报告18的FAIL；本轮actor起点相同不等于已解决跨后端或批调度数值变异。

## 3. Stop机制检验：与预期方向相反

| 历史探针seed/轮 | O更新前→后logp | O变化 | Z更新前→后logp | Z变化 | Z变化−O变化 |
|---|---|---:|---|---:|---:|
|90931/5|−.691406→−.753906|−.062500|−.691406→−.699219|−.007813|+.054688|
|90012/5|−.466797→−.480469|−.013672|−.466797→−.439453|+.027344|+.041016|

预设要求两个Z−O变化均≤0且均值<0；实际两个均>0，故历史Stop迁移门失败。原值头反而对这两个token降低得更多；其中一个Z还升高概率。不能把报告18“V=0使优势符号转负”直接改写成“真实更新一定压低Stop”。

**新32条训练数据没有任何解析后的Stop回合。** 两个历史Stop是已公开的事后迁移探针，不是本批同Stop样本的直接梯度干预；因此，本轮既不能证明也不能完整否定旧PPO崩落的Stop直接梯度因果链。

新批的答案内容token显示另一条有限线索：目标失败组原头86个答案区间token中73个优势为正，Z为0/86；目标成功24个token在两分支都为正。实际目标成功答案mean Δlogp为O+.00110、Z+.00601，失败组为O+.00265、Z−.00184。这里包括与答案区间重叠的标点/边界token，不冒称独立纯动作logit；其他成功组Z实际mean Δlogp仍为负，说明共享参数更新不能由局部优势符号单独预测。全分层与分母见[fresh signals](../docs/evidence/components/value_head_single_step_v1/Z/fresh_answer_signals.json)。

## 4. 保留的故障、实际成本与模型

原O完成16个b0基线、32 actor＋16同权训练态检查＋32 critic前向后，新的保存代码把DataProto合并storage上的不同dtype视图直接交给torch.save，报“view the same data as different types”。704-byte部分文件没有data.pkl或tensor payload；这是本次新序列化代码错误，不是原PPO错误。原O退出1、没有dispatch actor更新。

CPU复现该错误并验证逐tensor clone保留dtype/数值后，才进行一次受限修复：保留C原32条与O原16条基线，重新初始化同b0/空优化器/原小头，仅重算不可恢复的32 actor＋16检查＋32 critic，未重复任何已完成环境局或actor更新。O基线来自原O实例，明确记录在[baseline carry](../docs/evidence/components/value_head_single_step_v1/O_repair1/baseline_carried.json)。所有开始时钟、原日志、原失败、原调用计数保留；原“不重复打分计划”不判PASS。额外48 actor/32 critic计入总成本。修复O、Z最终分别exit0。

| 阶段 | 实际环境局 | 原生调用 |
|---|---:|---:|
|共享采样C|32|97|
|原O基线后保存失败|16|60|
|修复O，仅post|16|60|
|Z基线＋post|32|121|
|合计|96|338|

另有actor前向280个样本(含失败及训练前向)、critic96个、全局actor step2(每分支1)、critic step0、入库0。两分支每rank均恰好一次有限非零参数更新；grad norm O8.2524、Z6.2956。不能拿工程更新正常代替性能证据。

保留O/Z两个BF16 inference-only模型，每个7,509,346,928 bytes。初始化已核实输入/输出embedding实际同一Parameter且配置tied；只省略重复lm_head，检查了关键key/shape和文件header可读。未保存optimizer，不声称完整训练恢复或独立重载已验证；权重保留在远端，不公开上传。旧b0/r0与32.56GB恢复checkpoint未删除。

首次准备00:47:17不重置，GPU阶段最终01:33:48退出，含全部准备的原4小时截止04:47:17保持；公开证据整理后观察占用约83.95GB，本组件新增约15.42GB，低于16GB硬预算/15.8GB写停与项目85GB线。12小时总界仍11:38:27、累计新增严格<50GB；关闭时无本组件残留进程。

## 5. 决定

历史Stop迁移机制门失败；原初始critic解释不能当已证实根因。性能门只通过了1个条件成功的有限线索，区间跨0且基线存在数值变异，不扩大常规训练、不再追加本协议actor step。

下一候选是只评估：用已保留的b0/O/Z，在另行冻结的新32张地图上作前瞻性比较，逐局串行以减少批调度变异；不新增训练、不开精度/内核网格、不重放这16张Dev来选模型。新地图、绝对/相对门、调用/存储/截止须先冻结。该后继在本报告写作时尚未冻结/启动。

证据：[完整结果/配对区间](../docs/evidence/components/value_head_single_step_v1/result.json)、[冻结数据与筛选](../docs/evidence/components/value_head_single_step_v1/data_manifest.json)、[O完整信号](../docs/evidence/components/value_head_single_step_v1/O_repair1/training_signals.npz)、[Z完整信号](../docs/evidence/components/value_head_single_step_v1/Z/training_signals.npz)、[退出/存储](../docs/evidence/components/value_head_single_step_v1/closure.json)。公开数值包包含实际输入、所有96局及338调用的合成游戏证据、固定库与必要源代码；不含模型权重、完整私有runtime日志或私有头参数。
