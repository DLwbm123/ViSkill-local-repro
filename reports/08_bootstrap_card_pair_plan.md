# 下一步：参数训练与在线图卡的配对先导

状态：**方案已整理，待CPU预检，尚未启动**。参考用户分享的[下一步实验计划](https://chatgpt.com/share/6adc69d5-92a8-83e9-9233-f255cbaf0483)，并独立核对当前本地实现。分享中的建议本身不是执行授权；本计划处于本会话既有持续自主实验范围内，所有启动仍须满足项目科学与资源边界。

## 对上一建议的修正

应先测试同一方法中“参数更新”和“在线图卡上下文”的早期学习路径，再决定是否改轨迹表示、局部子目标或引入课程。单步图卡虽不能激活当前失败技能奖励，仍可作为训练输入；多步技能与非空库恢复不是所有学习诊断的前置要求。原始自由文本E和完整F仍PARTIAL；既有方向化变体只在限定组件范围通过，不能外推为原论文通过。

共享计划中动力学描述与检索排序问题已有本地结果：32对前瞻检查未达改进门槛；160保存状态的检索审计没有漏检有效替代。不重跑、更名扩样或继续搜索旧8条提示。新训练两臂统一显式使用准确动力学文字，作为描述一致性选择，不宣称其已证明提升。

## 两臂一次性冻结

A：普通全参数PPO、空库，保留轨迹与旁路self-distill原文，无正式入库或技能奖励。作为服务于主线判断的固定标准对照，不展开baseline调参。

B：同一b0和独立空库，现有方向化self-distill、完整语义和适用性guard、质量1/novelty.9/检索.8保持；合格技能在线入库，并供后续batch使用，技能奖励权重为0。

两臂同128张新训练地图，4批×32，mini8/micro1，rollout n1，训练temperature1，既有学习率与clip，无shuffle。Dev64张，训练前后greedy评价；最终theta4再跑独立32张train-probe，允许旁路蒸馏但无参数/库更新。未来Test256张只预留，本轮不调用模型。候选seed段与预算见configs/smoke/bootstrap_card_pair_v1.json，必须先核对真实棋盘及初始位置与历史/组间不重复，才能冻结为可运行协议。

主要解释：A检查少量参数更新能否改善合格多步技能源；B检查在线图卡是否实际进入后续训练。图卡若未出现/未注入，应报告未触达条件；单步曝光只能证明该范围的训练通路。单训练随机种子的4批结果不是稳定学习收益或最终收敛结论。

## 核对过的实现与待完成项

- 当前两个trainer都将save_freq>0作为正常保存必要条件；仍须关闭HF upload，因为上传分支也可触发_save_checkpoint。禁存可做，但不是完整F恢复实验。
- 普通入口使用基础trainer，无DistillCallback；技能入口使用技能trainer并初始化callback。A必须保留skill agent/provider与轨迹收集，不能简单把skill_system.enable关掉。
- 现有自蒸馏判断不区分验证阶段；必须显式接入阶段控制，Dev禁止蒸馏/入库/utility更新，最终probe禁止入库与optimizer更新。不能只加运行代码不读取的配置项。
- gen_batch_size优先于train_batch_size；两者都需显式为32，实际展开检查4批与128个seed。
- 旧wrapper硬断言只跑一步，必须新入口，不修改旧协议。
- 奖励放置与全局截断存在需要检查的边界；当前没有已观察到的触发证据。新运行应确认非零rm_scores对应response_mask=1，产生终止动作的输出仍在训练tokens中。
- 需要在同一存活trainer/cluster内完成theta4后Dev和probe，不能退出后再假定模型可重载。记录真实policy/library版本与同步证据，不能只靠step标签。

## 有限预算和停止

每臂最多1600模型调用、120分钟；两臂合计3200调用、4小时、新日志及小型证据2GB。原项目总截止1792858781.9046373（2026-10-25 00:19:41，Asia/Shanghai）不变，局部截止取更早者。仅物理GPU3/5，启动前实测可用显存和实际写占用；85GB保守停止线、旧检查点保护保留。

本轮不存新模型，全部前后评价与probe在进程退出前完成。因此结果不含最终可加载权重，也不证明独立恢复或断点续训。存储不足不能永久阻止这项窄范围学习诊断，但完整F仍需先解决新权重与非空库pair的保存预算。

两臂4批结束即停，不根据结果追加。若仅有单步技能仍如实保留；若有多步轨迹但语义失败，本轮不搜索新提示。B可旁路计算自然轨迹的L/f/m及lambda.8支持，不将其当真实奖励训练。只有以后另立奖励实验，才能声称奖励进入优化。任何准入指标都不得改写旧失败。


## 已冻结的实现与预检

用户已明确要求按计划启动并允许清理无用途大文件，新实验新增必须小于50GB。本轮保留更小的2GB新写预算、200MB停止余量，原85GB保守线和24h总截止不变。清理25个本项目安装下载归档，释放40,345,600 allocated bytes；基础/参考模型、旧检查点、数值结果与失败记录保留。torch2.8.0+cu128/sglang0.5.2环境导入正常。

CPU预检固定480个确切棋盘/初始位置（Train128/Dev64/Probe32/Test256），对已保存历史状态与既有开发seed重建共237个身份排除重叠；按固定升序候选池剔除4个重复，仅按身份去重，未依成功率或BFS选择。完整清单在docs/evidence/components/bootstrap_card_pair_v1/data_manifest.json。训练实际展开4批且sentinel不进入数据。

两臂配置通过bundled validate_config；5项CPU检查通过，包含真实processor+环境+官方loop子类的脚本fixture交互，证明Train/Dev/Probe蒸馏阶段行为与probe采样参数生效。这些fixture调用不是真实模型推理或PPO证据；GPU运行另行记录。首次CPU预检的numpy序列化错误与重复地图发现保留在协议，不改变历史实验。

新增代码为独立agent/dataset/trainer mixin与运行入口，不改旧smoke单步限制。现有FSDP同步路径仅增加默认关闭的返回记录；optimizer审计增加时间字段。真实训练将记录raw请求、生成/观测mask、终止输出、图卡处理、版本与更新、实际权重同步返回；同步回执不等于全量权重逐字节等价证明。Dev和Probe保持库不变，checkpoint保存入口直接拒绝调用；最后theta4探测在原存活trainer中完成。

准备起点1792830540.3041732纳入本轮4h墙钟上限，因此整个pair截止不晚于1792844940.3041732（2026-10-24 20:29:00 Asia/Shanghai），修复不重置。原项目总截止更晚但不延长本轮局部预算。协议现为FROZEN_CPU_PREFLIGHT_PASS；尚未凭预检声称训练已启动，实际启动读取state/components/bootstrap_card_pair_v1_receipt.json。


## 实际启动（2026-10-24 16:44）

已启动bootstrap_card_pair_v1，supervisor 4163385，run_token b1955d05cd5f487584cdeaffc76833ed；start 1792831455.7934191，A臂start 1792831456.4168003、deadline 1792838656.4168003。A/B顺序执行，B仍待A完成，不代表两臂都已训练。原生Ray两卡worker已开始初始化actor/critic，暂无训练结果。

实测启动保守基线68,431,892,480 bytes，独立写停止线70,231,892,480 bytes，即新增1.8GB停止、预留0.2GB余量，严格小于用户50GB上限；不保存新模型。pair硬截止1792844940.3041732，含准备阶段的4h预算；原项目24h截止不变。仅使用物理GPU3/5，启动前两卡空余均超过44,000MiB。监督进程已存活并持续按token+创建时间检查自身进程、中性argv、存储和时限，运行不依赖当前SSH或聊天连接。

日志logs/ppo_smoke/bootstrap_card_pair_v1_A.log与_B.log；状态state/components/bootstrap_card_pair_v1_receipt.json及_EXIT.json；真实输出results/ppo_smoke/bootstrap_card_pair_v1/A与B。数据清单与预检证据本地docs/evidence/components/bootstrap_card_pair_v1。唯一viskill 30分钟跟进将读取真实退出状态，不重启已完成batch或模型请求。


### 17:36 运行故障已原位修复

A训练前Dev曾在54次已返回请求/15条完整episode处停滞，确认为并发环境关闭调用进程全局pygame.quit等待。已修复RGB生命周期并通过6项相关CPU检查；只完成当前内存中唯一cleanup waiter，保留61010落洞失败原结果，无重复推理/动作/参数更新。现已恢复到75请求/20条Dev；原时钟、预算、4批与B排队顺序不变。详见09_bootstrap_incidents.md及原始栈/修复receipt。


### 18:29 A完整结束、B在资源恢复后启动

A四批及前后Dev/probe全部结束exit0，1047请求/288episode；Dev成功9/64→6/64、质量8/64→5/64，没有改善证据。唯一有效多步合同来自更新前theta0，不能归因为学习。B在首次GPU5显存门槛不足时未启动，随后按原门槛恢复，只运行未开始B。现在读取recovery receipt/EXIT优先，旧A输出和原资源失败EXIT保留，详见10报告。


### 最终状态：两臂均完成

A/B均完成4批及全部前后Dev/Probe，共576局2056调用，恢复EXIT B0/reasonnull。工程卡训练通路有证据，性能提升不判通过；最终结论以11_bootstrap_pair_results.md为准，前文启动/等待/运行中描述均为历史状态。不追加该协议。
