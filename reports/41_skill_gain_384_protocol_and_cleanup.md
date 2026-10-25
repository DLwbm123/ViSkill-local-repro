# 384局视觉技能增益：调用前冻结协议与清理记录

用户授权先执行第一步384局技能增益实验，保留所有正负结果并按事前门槛判断是否值得进入内化训练；没有授权本轮自动开始内化训练。另授权删除项目过时、不需要的大文件，并明确选择保留旧O分支唯一权重。本协议在本轮模型调用前发布，执行源码 `eaf85eaae9e223cc27bc6679dea477530aa9717a`。

## 数据、条件与重复

冻结基础b0和最近已核查的step1快照10张在线技能卡；模型不更新，库不蒸馏/入库/更新utility。64张新地图=32自然+32按原top1检索score≥.8筛出的地图。按预定新候选区间升序首额选择，不看模型输出；排除已记录历史/保留Test manifest和episode中的完整desc棋盘以及新panel重复。实际读取1531种历史棋盘，扫描103个候选；自然32中8个可检索，可检索层32个，合计40/64实际具备注入条件。不能把此排除范围说成核查了所有历史未记录的CPU候选。

源动作模式为down/up/right-right-down-right/left/right，共5种；所选图的top1技能分布为skill_0:9、skill_1:7、skill_2:7、skill_3:6、skill_8:11，其余24图不达阈值。库中重复动作模式不计成不同模式。

每图N无卡、V原真实卡、Blank同文字/图像槽位/尺寸的纯白参考卡；每条件两次独立引擎初始化的greedy评估，T0/top_p1，engine seeds42/43。共64×3×2=384完整episode。每重复的地图顺序提前随机固定，三条件顺序使用六种排列平衡。两次重复用于重启重复性检查，不是128个独立地图；安装后端没有请求级采样seed接口，不声称逐请求同随机数或逐token确定性。

使用原native GymAgentLoop、原动作解析/环境/检索阈值，最多5turn×512tokens、每turn最多5动作、prompt5000/response4000/context9000。Blank只替换参考卡的像素，保持原intro/布局。自然层未达检索阈值的V/Blank保持未注入，不强行填卡。库是只读共享资源，全部条件初始棋盘必须与冻结清单相同。

## 调用前门槛

完整384局和输入/库检查首先必须通过。随后同时满足以下四项才判值得下一步内化pilot：

1. 至少12张实际检索到技能的地图，在两次重复中均V成功、N失败；
2. 任一重复出现N成功、V失败的地图并集不超过4张；
3. 可重复增益覆盖至少3种不同源动作序列；
4. 其中至少一种长度大于1的多步序列。

门槛含义在源码和CPU边界fixture中提前固定。Blank的全部配对正负结果、路径质量、Stop/落洞均完整报告，不在看到结果后追加或放宽数值门。门是投入判断，不能代替统计显著性或参数学习收益证据。

报告自然32、可检索32和均衡panel64的分层分母，分别给两重复的V−N/V−Blank改善/退步，以及按地图聚类的95% bootstrap区间；两次重复不冒充更多独立地图。BFS仅用于事后路径质量评价，不给模型答案、选成功地图或改技能策略。

## CPU验收、预算与停止

实际native循环+processor三条件fixture已通过：V/Blank首轮prompt IDs、参考/环境图尺寸和grid完全一致，Blank像素全白，N无卡、V/Blank同源检索ID，初始环境与清单一致，蒸馏关闭。新增384完整合成门槛fixture验证≥12边界、11例失败、5例伤害失败、模式/多步不足失败；0实际模型/GPU调用。脚本式CPU环境fixture3局单独计，不混入384业务结局。

本轮准备起点17:14:06，4小时含准备硬截止21:14:06，20:59:06协作停止，重试不重置。单GPU3，1个serial引擎/重复，BF16/Triton文本/SDPA视觉/max_running_requests1。最多1920次模型生成请求、0self-distill、0optimizer/入库/utility更新。独立supervisor每5秒守时间、存储和自身进程，argv不含用户名/项目方法；不依赖30分钟监测充当硬限制。

清理后四根baseline=154,221,658,112 bytes；新增严格小于8GB，7GB停止写入，含日志/请求图像/缓存/证据。模型和源库只读共享，不复制权重。完整请求意图/返回/输出token、输入IDs/图像、轨迹、所有384正负结局留服务器；公开source/config/完整小型结局和生成输出，排除权重/运行令牌/批量输入和完整runtime。发生故障保留已完成局次和成本，不换图或重采样直到门通过。

## 已完成的大文件清理

锁定当前GPU服务器的本项目根目录，确认无进程消费者后，仅删除已完成PPO审计的派生 `PRIVATE_REPLAY_GENERATION.pt`：logical1,050,185,307 / allocated1,050,189,824 bytes，文件系统可用空间实测增加1,050,185,728 bytes。真实actor输入bank两份、原415条回复、重建源码和全部结果/失败均保留；这个CPU派生缓存不再供当前任务使用。

旧O唯一7.51GB推理权重按用户明确选择保留。b0、r0、最近step1完整model/optimizer恢复点、原始输入/结果/配置与历史唯一模型均保留，未哈希、未删除其他项目或共享运行环境。清理不计为新实验的科学正结果。

[冻结协议](../configs/reproduction/skill_gain_384_v1.json) · [64图/384顺序](../docs/evidence/reproduction/skill_gain_384_v1_prereg/data_manifest.json) · [CPU检查](../docs/evidence/reproduction/skill_gain_384_v1_prereg/cpu_check.json) · [清理回执](../docs/evidence/reproduction/skill_gain_384_v1_prereg/cleanup.json)

## 首次初始化失败与调用前修复

首次supervisor在17:31:41启动并于17:32:28退出1。首个环境已reset，记录器预分配计数1后因未创建request_images目录报FileNotFoundError；还没有写请求intent、没有进入模型generate，实际业务模型请求0、完整episode0。GPU模型初始化与约47秒运行成本确实发生，保留原失败、原源码eaf85eaae9e223cc27bc6679dea477530aa9717a及原时间/预算。

修复在引擎初始化前创建request_images/episodes/trajectory_images，计数记录移动到输入保存成功后；新增CPU检查实际Manager记录器与原native transport入口，stub返回3次，真实写入5张输入PNG与6条intent/result后检查通过，实际模型/GPU调用0。原三条件native循环fixture保持通过，未重选64地图或修改384顺序/门槛/T0参数。

只做一次明确的生成前修复，修复执行源码 `2d5cff664d7fbb62e1da934e43fe323dc9b56bce`。原失败位于initial_failure证据子目录，运行令牌不公开；同一准备起点17:14:06/21:14:06硬截止和原baseline均不重置，不重复任何已完成的业务模型局次。

## 第二重复前的显存门中止与仅续第二重复

第一重复192局全部完成，实际691次业务请求，全部intent/result匹配、0请求错误；64图的初始状态、检索ID及V/Blank首prompt IDs/尺寸/grid检查通过。worker在17:50:56切换第二重复前的60GB显存门处退出1，supervisor于17:51:06完成清理。第二重复尚未初始化引擎或生成请求。

失败时的具体free值与消费者未保存，不能把延迟释放确认为唯一根因。随后仅清理本轮已识别进程并确认0残留，GPU3实测72.83GB空闲，没有改动其他任务。改为每个重复一个独立worker进程，由supervisor在重复间完成自有进程清理，以消除同worker生命周期依赖。

本次只恢复repeat1/固定engine seed43，原192局及691回复保留、不重放。CPU预检查验证原192键准确等于repeat0冻结顺序、没有任何repeat1请求，并验证启动repeat0会被拒绝；真实模型/GPU新增0。恢复执行源码 `f4a8488147c1d9ff81c11ac2faf646189a7cd1c5`，门槛、64图、384顺序、T0、原17:14:06时钟/21:14:06硬截止/8GB预算均未改变。第二次故障与成本单独保留，不把192局当384局科学判定。
