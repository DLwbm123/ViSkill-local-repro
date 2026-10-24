# 起点支持匹配课程：调用前冻结协议，尚无模型结果

本独立对照测试训练组成：同24条新条件/中等起点数据，对照Q补8条单步数据，干预R补8条长起点BFS4+数据；每政策均32条、一次global actor step。动机是[CPU审计](26_retention_cpu_screen.md)观测到原C32最大起点BFS3，而新自然Dev9/16为4–6。不能据这一差异或一个退步例证明原因，也不能把旧报告25门失败改写为通过。

损失算子沿用已有答案重叠PG分子作为固定构建部件：full-generated GAE1/1与whitening、原全生成分母、zero baseline、同b0/空AdamW/LR1e-6/mini32/micro1/一个epoch/相同clip、entropy0/KL0、同5张卡。没有mask、学习率或seed搜索；数据组成及其奖励/whitening变化是干预的一部分，不声称只隔离长度标签。

调用前CPU冻结全新bank40（条件16/medium8/single8/long8）与Dev40（条件16/medium8/long8/single8）。候选分别400001..416384、500001..516384，升序首额，整board排除15080历史board、17058实际candidate-prefix seeds及全部reserved Test；80图互不重合。两政策共有24条精确样本、相同分支位置，只替换8条。全部40个b0采样结局保留；四类均至少1成功、40条均非空答案支持才训练，否则保留全部数据并停止，不补样本或扩大池。相比草案，在调用前收紧到四类正反馈门以保证两个训练分支都具有可用支持。

新Dev还含8个单步保持局次，避免替换单步训练后隐藏遗忘。固定原始b0/Q/R共同同顺序、串行batch1/agent worker1、T0：R条件质量≥6/16、成功/质量对b0各≥+2且对Q不降，条件配对退步各≤1；其他自然成功/质量≥4/16且对两参考各≥+2，配对退步各≤1；long质量≥2/8且成功/质量对两参考各≥+1；single质量≥4/8、成功/质量对两参考不降且配对退步为0。自然和single的Stop回合、含Stop局与hole均不得超过两参考。完整各组配对计数/区间都报告，这只是单训练seed的pilot投入门，不是稳定收益声明。精确机器可读门见[冻结协议](../configs/smoke/support_matched_curriculum_v1.json)。

CPU已验证实际40行collection/40×1 evaluation接口与_get_gen_batch路径、8个phase配置、共同24行的相同位置、GAE/subset映射、Worker MRO正确；新的答案mask函数与原C32的233个位置完全一致。0模型调用、0环境动作。40-case evaluator显式使用40局/200调用上限，不借用旧32-case fit。

最早研究准备05:22:50，4小时原截止09:22:50保持，global11:38更严。候选保守存储baseline84307259392 bytes（更早04:52观测），含随后全部准备；实际输入压缩测量后、调用前将设计草案上界调整为hard650MB/write620MB，绝对写停84927259392，仍低于85GB，未改变任何已有实验预算。新bank压缩写停70MB，两私有模型各225MB；首个参数更新前需至少500MB剩余写空间，覆盖两个artifact及50MB日志/公开证据，第二次更新前再次核算。首步前原钟至少剩1小时、第二步前至少45分钟。预测不足则停止，不删除任何旧模型或输入，不缩图/截断history。

最多collect40+eval120=160局、800次native调用；两global actor steps、0critic/distill/admit。完整计划FSDP actor前向232（两分支各pre32+训练32+post32+首次检查16，两次reload各业务2+额外2）。每个新私有模型都先在固定前两条共享保存输入上做独立reload门（max≤.05/mean≤.005 nats），两者通过才运行任何Dev；R更新前还核对24条共同样本与Q的实际pre分数。工程或科学失败保留成本与已完成步骤，禁止重复它们补救。

本文件是调用前协议，尚无采样、更新或性能结论。原始精确多模态输入及学习差分/schema私有保存，公开仅source/config、合成结局、过滤数值与元数据。旧报告25/21和所有工程失败保持原结论。
