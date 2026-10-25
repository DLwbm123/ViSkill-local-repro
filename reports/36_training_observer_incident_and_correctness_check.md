# 首批训练后的记录器故障，以及限定正确性核查

用户澄清，当前希望尽快确认原方法复现是否存在实现或配置错误，以便开展自己的实验；并非要求先把自训成功率追到84%，也没有要求现在设计新方法。已验证的发布模型216/256结果仍有效，当前重点收敛到原生训练更新与保存恢复。

## 必须保留的实际故障

official_frozenlake_ppo_v1于13:34:01退出。新增RunTracking记录器在首批原生actor/critic更新调用返回后，试图把Tensor指标直接写成JSON，抛出`TypeError: Object of type Tensor is not JSON serializable`。这是本次新增观察代码的错误；原CPU检查漏测了实际Tensor指标。错误发生在首个额外checkpoint保存之前，因此**更新后的活跃权重丢失，没有完整checkpoint可以无缝续跑**。

| 已保留事实 | 本轮实际值 |
|---|---:|
| 初始化完整验证 | 256局，34成功 |
| 首批训练采集 | 128局，20成功；这些是更新前行为 |
| 实际请求 | 1447 = 1405 rollout + 42 self-distill |
| 原生在线库 | 12条 |
| actor/critic原b0加载missing/unexpected/mismatched/error | 均为空 |
| 两模型实际初始参数dtype / tied embedding | FP32 / true |
| 保存的完整首步checkpoint | 0 |
| 首步正式loss/gradient数值和参数delta | 未保存、未知 |
| 本轮存储峰值新增 | 25,157,632 bytes |
| 闭环活动own残留 / argv违规 | 0 / 0 |

[完整故障审计及全部384局](../docs/evidence/reproduction/official_frozenlake_ppo_v1_failure/failure_audit.json)、[全部1447条生成回复](../docs/evidence/reproduction/official_frozenlake_ppo_v1_failure/requests)保留。不把原生调用返回等同于已核实非零参数更新，不把128局采集成功率当训练后的表现。这轮301步训练未完成；旧协议、失败、日志和成本不覆盖、不删除。

## 具体修复

记录器现在显式把Tensor detach到CPU并转为普通标量/列表，递归保留NaN/Inf标记；有限性检查也覆盖Tensor。若观测写入仍失败，优先尝试原生完整保存后受控结束，避免同类日志错误再次先丢失活跃状态。

针对真实根因的CPU回归已通过：Tensor梯度/损失、嵌套Tensor列表可写，原调用及返回保持，Tensor NaN触发受控停止；并复用了原配置、128条父类_gen_batch、async两角色RPC/MRO、HF保存构造器检查。0 GPU、forward/backward、env.step或真实optimizer更新。见[CPU回执](../docs/evidence/reproduction/official_frozenlake_correctness_v1/cpu_check.json)。

固定官方源的现有vagen/verl核心改动只有FrozenLake.close生命周期修复；训练算法文件未改，新增观察器单列。它有助于定位“自己改错算法”的边界，但不能仅凭源码相同就宣布所有运行时行为均正确。

## 调用前冻结的短程正确性核查

[official_frozenlake_correctness_v1](../configs/reproduction/official_frozenlake_correctness_v1.json)是保留原失败后的新工程尝试，**不是从丢失权重无缝恢复**。训练仍从原b0与空库、原数据/sampler与seed42开始，完整batch128/mini32、原GAE/fullgenerated PPO、actor/critic lr和FP32 AdamW均不改。固定只执行一批，保存完整actor/critic model、optimizer、RNG/scheduler、dataloader及配对在线库后停止；不进行第二批或301步性能试验，不按胜负补采样。

后续自动顺序固定为：

1. 原生一批训练和完整保存；实际指标须保留，actor/critic梯度有限且非零。
2. CPU读取事前指定的9个decoder矩阵小切片（首/中/末层的q_proj、o_proj、down_proj，各首2行×2048列，共36,864元素），对比原b0与保存actor HF导出，核查有限性和非零变化。不读全模型、不hash，不冒充全参数或critic delta验证。
3. 清理本轮own进程后，另起原生actor+critic，严格加载这份model/optimizer/extra及dataloader，保持库配对；只做恢复后完整256图验证，0新训练更新。此处检查加载和运行，不设成功率必须上涨或达到84%的门。

成功结论仅限这些工程边界。在线技能的蒸馏、入库、检索按实际计数报告；某批没有入库或短程成功率不涨，不自动归因为实现错误，也不由此放宽条件、选新seed。跨后端全分布一致性、长程稳定性和完整301步论文性能仍是不同的证据。

本阶段沿用用户已授权的本次150GB/四根210GB及原13:02准备起点，原24小时截止不变，父轮25MB等成本合计。另加当前短程准备起点起2小时的更紧截止，监督器按两者取严；不重置父时钟或新增存储基线。只有核查完整完成后才报告训练闭环通过；当前状态为CPU修复通过、GPU短程核查待启动。

干净执行代码冻结：`e42708bfd48016ff617fc28ff172be8a909c87a1`。原失败执行代码a3e18a5仍可定位，公开仓库不包含权重、optimizer、全运行日志或批量原始输入。


## 实际后续结果

限定一批训练、固定参数切片、完整保存及新集群原生重载均已完成，EXIT0；有限工程结论、实际108.90GB保存大小和全部成本见[报告37](37_native_training_correctness_result.md)。没有恢复301步长训练或启动新方法。
