Official full-batch FrozenLake training task has started and is initializing base actor/critic; first-batch time and complete recovery-checkpoint size remain unmeasured. See [report35](reports/35_official_frozenlake_training_protocol.md).

User authorized one official FrozenLake training exception:150GB additional/210GB four-root cap,24h unchanged. Original batch128/mini32 GAE/PPO from b0 and an empty online library is CPU-ready and preregistered; GPU training has not yet started. See [report35](reports/35_official_frozenlake_training_protocol.md).

Original full-training checkpoint storage requires a budget decision: about105.13GB tensor payload exceeds the current50GB per-experiment cap; no GPU training launched. The original algorithm/config remains intact. See [report34](reports/34_official_training_storage_feasibility.md) for the numeric check and specific resource proposal.

Official released-model full256 verification completed:216/256(84.375%),284 rollout dispatches and1 post-episode distill,0 updates/admissions. Initialization failure and its costs retained. Full training reproduction remains pending. See [report33](reports/33_official_released_frozenlake_check.md).

Official256 evaluation: first initialization failed before task dispatch because the observer inherited the synchronous worker; the released async worker is restored, original failure retained, and one repair is initializing under the same deadline. No performance result yet. See [report32](reports/32_paper_reproduction_reset.md).

按用户最新优先级回到官方论文/发布代码复现：已清理40.04GB退休权重，独立干净官方树冻结完整256图发布模型核对。此前方法诊断不作为原论文主结果复现。见[32](reports/32_paper_reproduction_reset.md)。

研究交接31：工程运行/推理恢复可用，稳定参数学习收益未建立。所有候选按原门关闭，现无待续跑GPU；后续先取得不同机制的数据/技能支持证据。见[交接报告](reports/31_research_handoff.md)。

动态检索CPU筛查30：全40局/132真实回合，long0/8新增有效命中机会，门未通过；0模型/环境操作，不启动该候选GPU。见[报告30](reports/30_turn_retrieval_cpu_screen.md)。

课程对照29：40局采集后发现意外2次蒸馏，程序退出；0训练/Dev，long成功0/8也不满足预定正反馈条件。本候选关闭。见[报告29](reports/29_curriculum_collection_failure.md)。

输入保存CPU验证完成（06:26）：C32实际容器290.59MB→私有gzip1 27.54MB，基本多模态读取通过，未做全文件字节核验。新的起点支持课程对照已beforecall冻结40条bank/40张Dev及共同24条位置，尚无模型结果。见[报告27](reports/27_input_container_feasibility.md)与[冻结协议28](reports/28_support_matched_curriculum.md)。

自然保持CPU筛查完成（2026-10-25）：原C32起点BFS均≤3，新自然Dev有9/16为4–6；唯一自然退步为BFS4，但初始动作正确，不能据此建立原因。微型fixture显示同参考KL/EWC首梯度为0，首步Adam的缩放效应取决于epsilon区域。未启动GPU；新课程对照仅设计，输入保存尚有容量阻塞。见[报告26](reports/26_retention_cpu_screen.md)。

答案区域损失分子试验完成（2026-10-25）：条件成功/质量b0 4/16→A 7/16，但其他自然2/16→1/16、hole5→6；A相对全生成Z条件成功仅净增1。冻结投入门失败，关闭当前协议，不追加mask/LR/seed搜索。96局/367调用、一个实际actor step，保留故障、换实例carry和私有恢复证明。见[报告25](reports/25_answer_region_step.md)。

恢复工程核对完成（2026-10-25）：私有BF16差分恢复与直接Z快照在两条固定保存输入的371个生成token上logprob差均0，实际8次actor前向、0训练/生成。只证明两输入的操作可用性，旧性能门仍失败。见[报告24](reports/24_bf16_delta_reload.md)。

前瞻性验证完成（2026-10-25）：新16条件地图b0成功/质量7/6，O3/3，Z4/4；Z低于b0且其他自然Stop回合4→10，预设投入门失败，关闭零值头单步性能线索。96局/364调用、0新训练，保留所有失败与成本。见[报告21](reports/21_prospective_value_head_eval.md)。

单步value-head对照（2026-10-25）：Z条件成功/质量0/8→1/8，O0→0；自然均1/8持平。只有1个改进、配对CI跨0；历史Stop迁移机制门失败，新训练批无Stop，未证原崩落原因。96局/338调用、两臂各一次actor step，保留新保存故障及修复成本，后续仅考虑新图只评估。见[报告20](reports/20_value_head_single_step.md)。

融合内核参考完成（2026-10-25 00:41）：同8布局关闭融合输出内核后，跨后端概率门仍8/8失败，误差与原设置接近。关闭该诊断支线，0训练更新。见[报告19](reports/19_kernel_score_reference.md)。

保存请求审计更新（2026-10-25）：8条概率全token门8/8失败，差异主要在think、动作最大误差.001381；原首批32条信号重建，两个Stop动作优势在原critic为正、V=0转负，尚无更新因果证明。接口与CPU分析事故、9次RPC/8次prefill/1个意外换行均保留，未训练。见[报告18](reports/18_recorded_backend_audit.md)。用户新授权12小时自主窗口至2026-10-25 11:38:27，旧总截止仅作历史，不据其截断后继；新增累计<50GB并保留原项目85GB保守线。

# ViSkill 本地无外部推理 API 复现

当前状态（2026-10-24 22:20）：固定库受控PPO两臂完整结束，共1408局/5135调用。B同库条件成功11/32→1/32，低于A最终9/32；自然多步成功6/53持平、质量6→4。冻结投入门槛失败，关闭本协议，不追加PPO或技能奖励训练。

**性能提升尚未建立。** Dev训练前均9/64成功、8/64路径合格；A训练后6/64、5/64，B训练后10/64、10/64。B自身增幅小，多步成功6/53→2/53，配对区间跨0，单训练seed不能称稳定收益。三步卡未被后续使用，实际技能奖励0；未保存本轮权重，非空pair恢复和完整F仍PARTIAL。原自由文本E失败保留。

- [固定库PPO最新结果与状态](reports/16_fixed_library_results.md)
- [当前固定库PPO协议与运行](reports/15_fixed_library_ppo.md)
- [完整四条件对照](reports/14_card_content_controls.md)
- [完整A/B结果与分项验收](reports/11_bootstrap_pair_results.md)
- [全实验汇总](reports/06_results_summary.md)
- [当前主线可行性](reports/07_mainline_feasibility.md)
- [P0证据与当前协议](reports/05_mechanism_p0.md)
- [完整状态与历史失败](reports/REPRODUCTION_STATUS.md)
- [真实PPO、检查点与重载](reports/02_ppo_smoke_test.md)
- [策略诊断](reports/04_distillation_diagnosis.md)
- [本地评估](reports/01_local_evaluation.md)
- [复现清单](reproduction_manifest.json)

仅使用GPU-server物理GPU3/4/5，最多3卡。当前12小时自主窗口截止2026-10-25 11:38:27，累计新增严格<50GB，另保留项目85GB保守停止线；旧起点与截止仅作历史。唯一30分钟监测为viskill-12。禁止自动300-step训练，医学迁移仍仅设计；显式修复、算法变体与原始失败分开记录。

公开源码快照：[ViSkill-local-repro](https://github.com/DLwbm123/ViSkill-local-repro)。公开交付当前按用户要求更新完整A/B结果；发布commit以交付回执为准。不表示原论文复现成功。


2026-10-24：独立16对三步卡条件化迁移已完整结束：同一b0空库1/16、真实图卡6/16，5改善0退步，16次三步注入；无训练或新权重。只支持狭窄适用集的即时图卡作用线索，学习收益仍未建立。见[第12报告](reports/12_multi_card_transfer.md)。

分享计划及P0复算已核对，后继优先独立四条件输入对照；重复Stop的正shadow支持使λ=.8训练继续暂缓。见[第13报告](reports/13_plan_review_and_failure_audit.md)。


四条件对照已全部完成：N/V/Blank/Text成功0/9/1/4（各16局），质量0/9/0/4；227请求，真实卡达到V−N与V−Blank预注册投入门槛。决定准备有限固定库PPO及权重×上下文交叉评价，尚未启动新PPO；学习收益仍未建立，技能奖励保持0。见[第14报告](reports/14_card_content_controls.md)。


2026-10-24：独立fixed_library_ppo_v1已启动，使用新的128训练/32条件Dev地图、原已开发自然Dev64，进行权重×上下文交叉评价。原bootstrap实验已结束，不追加第五批；新实验不保存权重，不声称独立恢复。
