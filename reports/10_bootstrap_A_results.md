# A臂完整结果与B臂资源恢复

A普通PPO+旁路蒸馏已完整正常退出：四批×32训练、64个Dev训练前后评价、32个最终theta4 probe，共288完整episode、1047模型调用和1047份请求输出。实际run_ppo返回，4484.96秒含此前环境关闭停滞；无新模型checkpoint，最终theta4未保留。B仍须完成后才能给配对结论。

| 阶段 | episode | 成功 | 质量合格 | 多步任务 | 多步成功 | 多步质量合格 | 有效多步蒸馏 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train（theta0–3） |128|20|14|100|10|5|1|
| Dev-before（theta0） |64|9|8|53|6|6|不调用蒸馏|
| Dev-after（theta4） |64|6|5|53|3|2|不调用蒸馏|
| Final probe（theta4） |32|4|4|25|1|1|0|

Dev成功和总质量指标均为1对改善、4对退步，其余匹配状态不变；本预算没有观察到开发集改善证据。四批单训练seed结果不证明PPO一般无效，也不代表最终收敛表现。Train各批地图不同，不能按逐批均值变化推断学习趋势；旧128采集也不能充当该组前评价。

Train14次蒸馏中10条有效，其中唯一多步为seed60024、BFS3、down/down/down，**来自首批theta0，尚未参数更新**。它证明现有base在一个新棋盘上能产生有效三步直线合同，不能归因于本轮PPO学习，也未正式入库（A库按设计保持空）。最终theta4 probe唯一质量合格多步仍蒸馏失败。不得把“出现1条多步”写成训练已改善多步技能来源。

Actor/critic共64条实际optimizer审计记录（各32）全部finite且采样参数delta非零；48条实际权重同步await返回记录保留。回执证明执行了对应同步流程，不证明全量权重逐字节一致。没有技能注入/正技能奖励/非空库恢复结论。

## B臂未启动阶段的显存门槛与恢复

初始supervisor在A exit0后检查B启动资源时，GPU5空余43,754MiB，低于预先44,000MiB门槛，因此退出且B目录、模型调用和更新均未建立。原始EXIT保留，reason不改写。未改变或终止其他作业。

后续一次实测GPU3/5空余分别49,813/64,919MiB，符合原门槛。仅恢复尚未启动的B，不重复A、不更改模型、数据、优化参数或GPU选择。使用独立recovery receipt/EXIT，原pair起点、原baseline68,431,892,480、write_limit70,231,892,480、原pair20:29截止不变。B的2h额度还保守计入资源等待，从初次资源检查退出记录计至19:59:37，不重新给满2h。

B恢复supervisor67233，token113693700e1746d1b49fb19bf5f0652e，实际start1792837774.980717（18:29:34），deadline1792843177.4530435（19:59:37）。已进入原生Ray actor/critic初始化。未来先读bootstrap_card_pair_v1_recovery_EXIT.json，再结合原A EXIT与最终result.json，不将原资源故障当成B已运行或最终pair已结束。

证据：docs/evidence/components/bootstrap_card_pair_v1/A/arm_review_summary.json、A/COMPLETE.json、原始EXIT及recovery_receipt。完整原始288episode、请求、梯度和同步记录留在服务器results/ppo_smoke/bootstrap_card_pair_v1/A。本轮尚未公开。


最终更新：B也已完整exit0；配对结论见11_bootstrap_pair_results.md。本文件的“B待完成/初始化”是当时历史记录。
