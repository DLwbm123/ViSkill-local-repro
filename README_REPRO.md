# ViSkill 本地无外部推理 API 复现

当前状态（2026-10-24 18:52）：A/B各4批×32全参PPO均已完整结束，共576局/2056模型调用。B在线库5条（4单步、1三步），后续10条训练样本实际使用单步图卡，限定工程通路已验证。

**性能提升尚未建立。** Dev训练前均9/64成功、8/64路径合格；A训练后6/64、5/64，B训练后10/64、10/64。B自身增幅小，多步成功6/53→2/53，配对区间跨0，单训练seed不能称稳定收益。三步卡未被后续使用，实际技能奖励0；未保存本轮权重，非空pair恢复和完整F仍PARTIAL。原自由文本E失败保留。

- [完整A/B结果与分项验收](reports/11_bootstrap_pair_results.md)
- [全实验汇总](reports/06_results_summary.md)
- [当前主线可行性](reports/07_mainline_feasibility.md)
- [P0证据与当前协议](reports/05_mechanism_p0.md)
- [完整状态与历史失败](reports/REPRODUCTION_STATUS.md)
- [真实PPO、检查点与重载](reports/02_ppo_smoke_test.md)
- [策略诊断](reports/04_distillation_diagnosis.md)
- [本地评估](reports/01_local_evaluation.md)
- [复现清单](reproduction_manifest.json)

仅使用gpu-host物理GPU3/4/5，最多3卡；当前项目存储保守停止线85GB，绝对新增占用不得超过100GB。原始总时钟起点1792772381.9046373、截止1792858781.9046373保持不变。仅保留viskill这一项30分钟监测。禁止自动300-step训练，医学迁移仍仅设计。所有显式修复、算法变体与原始失败分开记录。

公开源码快照：[ViSkill-local-repro](https://github.com/DLwbm123/ViSkill-local-repro)。公开交付当前按用户要求更新完整A/B结果；发布commit以交付回执为准。不表示原论文复现成功。
