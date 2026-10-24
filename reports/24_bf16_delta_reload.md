# 两条固定输入恢复核对通过，尚无新的学习结果

固定的已保存Z模型由保留b0和私有BF16差分恢复后，在真实生产actor路径上的两条保存输入，与直接载入Z快照的生成位置logprob完全相同：seed130014共316个token、seed130007共55个token，两个样本的最大/平均绝对差均为0，达到事前`.05/.005` nats门槛。实际HF加载的missing/unexpected/mismatched/error列表均空，embedding与输出头保持同一Parameter。只支持这两条输入的实际推理恢复可用性，不证明全权重字节身份、FP32/优化器恢复、未见任务效果或学习收益。

两政策都使用生产FSDP、FP32 master/BF16 mixed precision、fused kernels、micro1，actor-only角色跳过未使用的rollout初始化。输入直接取报告20的C/shared_batch.pt最前两条，包含原处理后的多模态张量，无图像重建或按结果选样。每政策2条业务评分，加两个rank各1条首次train-mode alignment检查，总计8次实际actor前向；0生成、环境动作、critic前向、优化步骤或新模型快照。全部371个生成token都纳入比较。

首次准备03:52:49，含准备1小时截止04:52:49不重置，修复后两政策04:04:34退出。原尝试因新自定义类名Worker与colocation框架按名称选择MRO基类冲突，外层构造缺少config/role，在模型初始化与评分前失败。原full目录、退出码1及成本保留；原全程exit0合同为false。重命名LocalActor，并在CPU实际selector先复现旧冲突、再确认正确基础Worker后，仅作一次评分前修复，没有重播任何已完成分数。初次CPU把可选mixed_precision段当作必有字段的断言错误也保留；核实实际dtype=bfloat16后只修正检查，未改模型设置。

组件100MB硬预算/90MB提前写停，绝对写停84,229,737,088 bytes与项目85GB、总窗口11:38:27取更严。实际运行峰值84,142,276,608 bytes；04:23关闭记录总84,142,272,512，相对准备基线新增2,535,424 bytes，归属残留进程0、命令行违规0。原b0/r0、O/Z完整推理快照及32.56GB恢复checkpoint均保留。私有162.57MB学习参数差分及参数schema继续留在服务器，不作为数值证据公开。

此核对结束恢复工程准备，不再扩展codec、样本或数值门。损失区域机制试验仍需另立冻结协议；零值头前瞻性性能门和历史Stop机制门仍失败，不以恢复通过改写。若后继成立，其最早准备时钟仍从03:22:49损失审计计入。

[完整两样本结果与成本](../docs/evidence/components/bf16_delta_reload_v1/result.json)，[关闭记录](../docs/evidence/components/bf16_delta_reload_v1/closure.json)，[存储格式准备](23_bf16_delta_feasibility.md)，[损失支持审计](22_actor_loss_support_audit.md)，[前瞻性负结果](21_prospective_value_head_eval.md)。
