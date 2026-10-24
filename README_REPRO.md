# ViSkill 本地无外部推理 API 复现

当前状态（2026-10-24 15:53）：原始自由文本E仍PARTIAL；方向化显式变体仅通过限定组件检查。一个全参PPO batch和独立模型重载已完成，但入库0，完整F仍PARTIAL。

P0数据链路检查通过。固定128条采集已完成：15成功、8质量合格、5策略有效、3单步技能入库；无多步技能通过。追加96次有限对照也均未达到各自门槛，未追加PPO。dynamics_wording_v1也已完成64完整episode：原/准确描述成功2/32与3/32，质量合格均2/32，未达预注册改进门槛；此提示支线关闭。唯一30分钟跟进有效。

- [完整结果汇总](reports/06_results_summary.md)
- [P0证据与当前协议](reports/05_mechanism_p0.md)
- [完整状态与历史失败](reports/REPRODUCTION_STATUS.md)
- [真实PPO、检查点与重载](reports/02_ppo_smoke_test.md)
- [策略诊断](reports/04_distillation_diagnosis.md)
- [本地评估](reports/01_local_evaluation.md)
- [复现清单](reproduction_manifest.json)

仅使用gpu-host物理GPU3/4/5，最多3卡；当前项目存储保守停止线85GB，绝对新增占用不得超过100GB。原始总时钟起点1792772381.9046373、截止1792858781.9046373保持不变。仅保留viskill这一项30分钟监测。禁止自动300-step训练，医学迁移仍仅设计。所有显式修复、算法变体与原始失败分开记录。

公开源码快照：[ViSkill-local-repro](https://github.com/DLwbm123/ViSkill-local-repro)。本公开更新包含P0、128条容量诊断、96次后继对照，以及dynamics_wording_v1完整结果与汇总，不表示原论文复现成功。
