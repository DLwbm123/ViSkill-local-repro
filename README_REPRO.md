# ViSkill 本地无 API 复现

官方仓库基线commit `7acaff880871b8d898d91c515c151080eabcba20`，本地分支 `repro/local-no-api`。不自动推送或发布。

2026-10-24最新：A/B审计PASS；C独立环境的导入、BF16/SDPA、FlashAttention内核前后向、GPU3/4/5 NCCL和SGLang真实图像→动作检查通过；D三臂60个episode完成，成功5/20、3/20、20/20，仅用于工程验证。E仍PARTIAL：reward/novelty显式修复通过，基础模型自生成策略存在语义错误；固定训练种子1–4的贪心和官方训练采样均未产生成功轨迹。F方向化变体已完成真实PPO更新及模型保存，但训练入库0、独立重载已成功，仍PARTIAL；G/H未完成。

仅使用pro5000 GPU3、4、5，最多3张；本项目新增实际存储≤100,000,000,000字节，85GB保守写入停止线。单个独立实验≤24小时；本轮准备及修复共用的原始6小时窗口已于2026-10-24 06:19:41届满；所有运行此前已退出，不重置时钟或续跑本轮。cold_start=false、distill=self；禁止外部推理API、遥测和自动上传；不启动完整300-step训练。

远程项目`/data/local/ViSkill`，独立环境`/data/local/envs/z0`。模型固定revision；不修改系统CUDA或其他环境。SGLang0.5.2已固定兼容依赖，运行采用Triton文本注意力/SDPA视觉注意力，区别于官方FA3默认；退出时存在已记录的非致命CUTLASS缓存警告。

最新E诊断：图像确实进入视觉编码器；放大图像、取消前缀、去示例均未解决原始策略语义问题。基础模型的精确棋盘位置和逐帧动作识别错误，但动作复述/计数正确。显式三阶段自蒸馏生成正确fixture动作并完成一次四轮复用，属于非原始推理流程，仍遗漏路线适用条件。反例棋盘在score=.8时检索到该技能但建议路径第一步落洞；补充条件文字后实际policy仍不能可靠遵守。不能据一次成功放行PPO。

用户要求重新分析并推进后，已完成有限因果诊断和适用性反例，详见[自蒸馏诊断](reports/04_distillation_diagnosis.md)。固定训练seed1–8真实policy检查已结束：2成功、1质量合格并入库，重载后复用8例、成功2例、技能使用3次，utility=.5355。修复了位置NumPy标量的JSON保存错误；原失败和未完成索引保留，未重采集/蒸馏。语义仍FAIL：seed6目标在下方一格，策略却说右下角。相对位置fast/slow处理器对照均0/8正确，不能靠换处理器解决。原始24小时时钟保持，无活动实验。一次核验反馈纠错及2×2输入对照也已完成：保留旧答案时有图/无图均原样复述错误；去掉旧答案仍混淆坐标/方位，全部未入库。后续转向结构化字段、语义核验与执行前适用性，不再盲目改写同一fixture提示；所有偏离原始算法的候选单列。旧提示变体停止决定不再阻止用户新授权的有假设诊断。

后续结构化JSON候选也未通过：Down1对应的目标位移被写成[-1,0]而非[1,0]；FP32/eager对精确相同输入输出完全一致。校验器6项测试通过，错误候选被拒绝入库。下一项仅对现有released权重做相同请求的固定对照，尚未运行，不代替base验收。

checkpoint与技能库配对修复为默认关闭的显式选项，6项CPU持久化测试通过；使用的是dummy模型文件，尚未证明真实actor/critic参数保存与恢复。D/E全部通过前禁止PPO。

唯一半小时监测为`viskill`。准确交接见[状态报告](reports/REPRODUCTION_STATUS.md)末尾；完整结果见[本地评估](reports/01_local_evaluation.md)、[组件测试](reports/03_component_tests.md)、[显式修复](docs/05_explicit_component_repairs.md)、[环境记录](docs/03_environment_setup.md)及[manifest](reproduction_manifest.json)。

最新实质进展：方向化策略变体已通过实际官方agent loop及原生SGLang组件闭环，2条技能保存重载、3次真实技能复用均成功，utility/use_count更新已核实。原始自由文本E的失败结论保留。详见reports/E_VARIANT_GATE.json；仅该显式修改后的候选进入1步全参PPO smoke准备，F尚未运行，不能称原论文复现完成。



最新PPO：真实actor/critic backward和非零参数delta已确认；本轮入库0，完整闭环仍PARTIAL。重载验证已成功，但恢复的是空技能库。见[实验报告](reports/02_ppo_smoke_test.md)。
