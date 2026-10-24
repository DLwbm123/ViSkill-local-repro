# 本地三臂验证（运行中，尚未验收）

协议在查看任何本轮VLM结果之前固定为`configs/low_resource/local_eval.json`：A base无技能、B release无技能、C release+发布技能库；每臂同一20种子10001–10020，最多5turn，每turn最多5动作、512新token，greedy、seed42。仅用于数据流检查，不声称复现论文成功率。三臂均不更新权重或技能库，不做cold start；训练期self-distillation另由E/F验收。

使用官方FrozenLake、parser、skill provider、feature/retrieval与prefix reward函数；C在system内容后追加视觉skill再加入初始观测，保持官方图像顺序。缺失视觉卡直接报错，不使用纯文本fallback。技能库按发布metadata只读加载，无evaluation utility更新。原始动作前缀reward偏差保留，逐episode分别记录outcome、format、prefix和total。

运行适配：Transformers4.56.1直接离线推理、BF16/SDPA、batch1、GPU4，重新tokenize完整历史；与官方SGLang异步增量token路径有差异。官方loader同样使用AutoProcessor默认值；当前版本发出fast image processor默认变更warning，记录并在三臂保持一致。图像使用processor原生缩放，每turn记录image_grid_thw与prompt/response token数；context上限9216，超限报错而非静默截断。Greedy为本轮固定验证设置，不据held-out结果调参。

A已完成20/20且EXIT0（下表）；原先后台启动，独立guard共享三臂与重试的6小时原始时钟（1792779344.2749124开始，1792800944.2749124截止），项目四处目录85GB停止写入线；最大发生存储仍严格受100GB总预算限制。B/C尚未启动，release仍下载中。

远程日志`/data/local/ViSkill/logs/local_eval/A.log`；receipt/EXIT在`state/local_eval/`；逐episode JSONL在`results/local_eval/A.jsonl`；初始和逐轮图像、响应和状态在`artifacts/local_eval/A/`。只有完整20行、summary与EXIT0、合法动作/环境交互证据同时具备才评估A是否通过。成功率、长度、invalid rate、retrieval、token、wall time与显存尚未完成汇总。

## A臂实际结果

| 指标 | Base，无skill |
|---|---:|
| episode | 20 |
| success rate | 5/20 = 25% |
| mean episode turns | 3.35 |
| mean primitive actions | 2.35 |
| invalid action turns | 21/67 = 31.34% |
| retrieval fraction | 0 |
| mean / max prompt tokens | 643.48 / 1068 |
| mean response tokens | 43.93 |
| wall-clock（含加载） | 83.84秒 |
| peak GPU allocated / reserved | 7,656,217,088 / 7,761,559,552 bytes |
| process max RSS | 4,912,660 KiB |

20个episode均已按terminal或5turn上限收尾，EXIT0；5次成功、5次落洞终止、10次到turn上限。所有21个invalid turn仍有正确标签格式，但模型输出了环境不允许的动作：Stop 9次、Done 6次、Forward 6次。因此不是把“格式正确”当成“合法动作”，也没有将这些输出转成合法动作来抬高成绩。模型存在过早宣称到达目标和方向词不符合动作词表的问题；这是本次固定协议下的负结果，不据此调整提示词/测试种子。

seed10003的初始图像已目视检查：玩家在(2,0)、目标在右侧(2,1)，模型Right动作成功；模型同时声称下方有洞，但图中下方是冰面，说明动作成功不代表其文字推理正确。该例仅验证图像→processor→本地生成→parser→环境动作→实际成功的数据流。原图和完整响应见`docs/evidence/local_eval/A_10003_initial.png`及`A_observation.json`。

记录字段澄清：官方env.step将info.actions改为实际执行动作，A原始JSONL中的parsed_actions因此实际表示executed actions。原始A.jsonl保持不动；A.annotated.jsonl从原始响应调用同一官方parser补充parser_actions和executed_actions，无重新推理、无指标修改。后续B/C直接记录两种字段。

A基础pipeline检查PASS；阶段D整体仍PARTIAL，B/C未运行，视觉skill注入尚待C臂证明；这些20例只作工程验证，无统计比较或论文复现数值结论。

## 三臂最终结果（固定协议，各20例）

| 指标 | A base无技能 | B release无技能 | C release+发布技能 |
|---|---:|---:|---:|
| 成功数 | 5/20 | 3/20 | 20/20 |
| 平均turn | 3.35 | 2.70 | 1.00 |
| 平均实际primitive actions | 2.35 | 9.20 | 2.05 |
| invalid action turn rate | 31.34% | 77.78% | 0% |
| 检索注入比例 | 0 | 0 | 100% |
| 平均检索score | — | — | .99590 |
| 平均prompt/response token | 643.48/43.93 | 608/33 | 519.85/24.75 |
| 含加载wall秒 | 83.84 | 70.19 | 25.96 |
| peak GPU allocated GB（十进制） | 7.656 | 7.653 | 7.601 |

三臂EXIT0和20行完整episode证据均已保存，D工程pipeline PASS。不同wall不能直接解释为模型速度差：磁盘cache/加载耗时与turn数量不同。B首次尝试仅加载后发现缺少chat_template.jinja，在任何生成前退出1；补齐固定revision的1017字节原文件，失败目录state/local_eval/failed_B_template_missing完整保留，重试共享原时钟，没有重跑已有episode。

B全部54轮parser输出相同序列`left, down, down, down, v player on goal`；最后词不在动作词表。若环境在遇到该词前已终止，未执行的suffix不会导致该轮invalid；因此42/54 invalid而非54/54。最终3成功/9落洞/8上限。没有替换checkpoint或调prompt。发布config的Transformers版本为4.56.1，与本环境一致；发布processor及模板已原样使用；lm_head/embedding的三个小切片相同且有限，仅为低成本排错，不宣称全权重一致或完全排除实现差异。

C使用15个不同skill，最低score .96626，全部高于.8。发布库265条有seed元数据，范围88–9942，与本次10001–10020无seed重叠；这不等于没有几何等价任务，也不证明广泛泛化。近满分检索覆盖是解释20/20的重要条件。seed10001已目视检查：检索卡轨迹up,right,up对应当前相对目标位置，真实成功；卡片strategy文字却含down/left，与示范动作不完全一致。实际有效信息可能主要来自动作/轨迹图，现有三臂不能隔离文字strategy贡献。

D的60例结果全部保留。本轮结果支持本地视觉技能读取/注入/交互链路，不能代替PPO学习、online库演化或完整论文数值复现。B/C运行时远程核心源码仍为原始路径，之后才部署显式组件修复。
