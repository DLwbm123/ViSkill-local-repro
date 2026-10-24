# BF16差分格式可行：162.57MB私有权重归档，完整模型恢复尚未验证

为在85GB项目线内保留后续主线试验模型，CPU只对一个固定的已保存Z模型测试了相对保留b0的BF16 uint16 XOR编码。**824个张量完成键/shape/dtype覆盖，694个写入差分数组，归档162,568,984 bytes，约为原7.51GB推理权重的1/46。** 这是存储格式的工程可行性，不是方法性能、完整模型恢复或FP32/优化器续跑证明。原O/Z和b0/r0/32.56GB恢复checkpoint均未删除。

输入为已保留b0和已保留Z，均只读。用安装版Qwen2.5-VL官方checkpoint_conversion_mapping映射旧键到当前键，要求全部824个目标键/shape/BF16 dtype一致；tied lm_head别名沿用已验证共享关系。固定一种NumPy/ZIP deflate6 XOR格式，没有codec搜索、模型前向或新训练。处理3,754,622,976个BF16元素，其中56,661,072个比特值发生变化；这个计数用于编码，不是全模型文件字节身份校验，也不能用于解释性能。

小型CPU bit-roundtrip、128-byte写停fixture及带共享embedding的微型参数容器apply回归均通过。全归档只做一次成员列表可读性检查，未重读全部数组验证、未计算hash。尚未把完整真实模型由该差分恢复后运行，因此full_model_roundtrip_or_reload_verified仍false；下一步若需要用于训练模型留存，必须先做至多两条固定保存输入、同生产actor路径的有界恢复前向核对，0生成/0环境动作/0优化步骤，并保持旧数值门。

首次准备03:33:25，20分钟含准备截止03:53:25不重置，CPU结果03:39:53完成。固定1个模型，300MB硬预算/250MB归档写停；超时、键/dtype或大小失败均停止。新增归档是**学习参数，始终按私有模型权重处理**，连同参数schema保留远端，绝不把它当numeric evidence上传。此处仅公开代码、配置和不含参数值的汇总。

若后续独立actor-loss区域机制试验成立，它的最早准备时钟仍从03:22:49的损失审计计入，不能用编码/核对拆名重置科研时钟。当前没有新训练启动；零值头性能门与历史Stop机制门保持FAIL。编码可行不构成重启已关闭路线的理由。

[格式汇总](../docs/evidence/components/bf16_delta_feasibility_v1/result.json)，[损失支持审计](22_actor_loss_support_audit.md)，[负向前瞻性结果](21_prospective_value_head_eval.md)。
