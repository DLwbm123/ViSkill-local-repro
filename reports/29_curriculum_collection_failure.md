# 课程对照在采集后因协议执行错误关闭：40局保留，0训练

本轮没有完成Q/R对照，不能评价支持匹配的学习效果。原始b0完成全部40局后，采集器检查发现两次意外self-distillation并退出；未保存精确processed bank，未执行任何actor评分/更新、模型保存或Dev。原始collect exit1和所有失败证据保留，不重采样或将重建输入冒充原始输入训练。[调用前协议28](28_support_matched_curriculum.md)保持原门与原预算。

错误来自配置继承：训练阶段仍为`distill.mode=self`，而禁止蒸馏的断言在全部轨迹生成后才执行。实际134次调用包含132次rollout和2次distill，违背冻结的distill=0要求。两次蒸馏都因relative goal condition mismatch无效；未运行入库回调，最终5卡库与原库JSON值相同。不是数据容量不足或科学门拒绝导致的程序退出。

| 完整原b0采集 | 局数 | 成功 | 最短长度质量成功 | Stop回合 / 含Stop局 | hole局 |
|---|---:|---:|---:|---:|---:|
| 条件 |16|8|6|1 / 1|2|
| medium |8|2|2|0 / 0|3|
| single |8|5|5|2 / 1|1|
| long |8|0|0|0 / 0|2|
| 全部 |40|15|13|3 / 2|8|

这些是完整训练采集的事后描述，不是heldout结果或两方法的比较。另一个独立限制是long成功0/8，已不满足调用前规定的每类至少1成功；即使没有上述配置错误，也不应继续两臂训练。实际bank_ready门没有被执行或保存，40条非空答案门未获得实际batch验证，不把它们写成通过。不能从8局推断原模型普遍不能处理长任务。

后续CPU核对保留全40局，不只选择两次蒸馏或失败案例；核对冻结seed、起点、类别和请求计数。新配置显式禁用蒸馏，真实`_should_self_distill`方法的小fixture确认旧self会进入、新disabled不会进入。这个修复仅用于防止未来重复错误，本轮未重新启动。完整多模态processed batch已随进程退出丢失，原始请求/token/合成PNG/轨迹仍在；本轮不做重新生成或重建后训练。

supervisor从06:57:34运行至07:02:27，最早准备05:22:50及原09:22:50截止均保留。原峰值84,365,852,672 bytes；07:33:49关闭观察84,365,950,976 bytes，相对保守候选参考新增58,691,584 bytes，campaign新增15,874,142,208 bytes。资源守卫保持ACTIVE、该作业own-token+ctime残留0、argv违规0。关闭后导出292个文件3,220,189 bytes，导出占用计入后续campaign统计。旧模型、输入和所有失败受保护。

公开[完整记录与失败核对](../docs/evidence/components/support_matched_curriculum_v1/failure_audit.json)、[原退出](../docs/evidence/components/support_matched_curriculum_v1/EXIT.json)、[成本/状态](../docs/evidence/components/support_matched_curriculum_v1/result.json)、source/config及全部40局/134请求/合成图像。没有新权重、private schema、原始多模态容器或完整运行日志。精确before-call协议已在先前公开提交保留，不用程序失败掩盖0/8正反馈限制，也不将工程失败当作支持匹配方法的科学否定。

本候选关闭，不补seed、不扩大池、不救援当前对照。下一步先判断如何获得可用的长任务技能支持；任何不同机制必须另立假设、真实控制和冻结预算，不能仅为继续运行而再采同一任务。
