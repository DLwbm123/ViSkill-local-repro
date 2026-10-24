# bootstrap_card_pair_v1：并发RGB环境关闭停滞与原位修复

## 已观察到的故障

2026-10-24 17:26检查发现A臂仍停在训练前Dev：54次模型请求全部返回、15条episode已写盘、没有任何optimizer更新；最后一次输出距检查约40分钟。独立监督进程仍存活，未到存储/时间限，不能据进程存活称其健康推进。

读取该实验自己Ray AgentLoopWorker的线程及异步任务栈后，确定唯一未完成的episode为Dev seed61010：已执行right，落洞，动作/位置/结果已在内存中；协程停在FrozenLake.close→asyncio.to_thread→Gymnasium FrozenLakeEnv.close，工作线程位于pygame.quit。其他七条同批任务已结束。模型推理结果均已返回；没有依据将该停滞归因为GPU权重、优化器或策略性能。

## 根因修复与检查

Gymnasium的close会退出进程共享的pygame/SDL运行时，不适合在并发RGB环境的线程池中逐episode调用。当前封装始终创建rgb_array底层环境，修复为只释放该环境的window/clock和实例引用，进程退出时再由进程结束回收共享运行时；不在episode结束时执行全局pygame.quit。

新增并发CPU检查：八个真实RGB环境并发reset，关闭其中七个后，最后一个仍能渲染和成功执行两步；同时禁止调用底层全局close。加上既有phase/token/库控制检查共6项通过，包含真实处理器+环境的脚本fixture交互，0真实模型调用。没有重新搜索提示或改变准入/奖励门槛。

## 当前进程原位继续

未重启训练worker、未重复54个请求、未重复任何环境动作、未改原输出。对唯一属于本run token的AgentLoopWorker加载修复后的close方法，仅为seed61010已结束episode释放RGB引用，并在原event loop完成那一个已确认卡在全局退出的cleanup waiter。原pygame.quit底层线程可能仍等待至worker退出；不声称该底层调用已经返回。修复操作及token/PID/阶段断言见scripts/repair_bootstrap_close.py及对应receipt。

原位处理后，61010以原来的落洞失败结果写盘；实际模型请求数继续由54增加到75、完整Dev episode由15增至20，证明管线重新推进。此时仍无optimizer更新，B仍按原计划等待。后续进程从已修复源码启动。

本次修复继续共享首次receipt、A臂18:44:16截止、pair20:29:00截止、新增2GB预算/1.8GB写停止线和原项目总截止。没有追加样本或训练批次，也没有放宽完整F验收。

## 诊断边界

只读诊断期间，一个辅助Ray查询进程因错误使用“本driver未注册的actor handle”接口退出；另有loopback地址解析与JSON numpy标量的辅助查询错误，均发生在独立诊断driver，无模型调用或实验worker重启。成功诊断改为读取GCS提供的namespace，用已命名的本run actor读取栈。所有原始栈和修复记录保存于docs/evidence/components/bootstrap_card_pair_v1/incidents。此辅助故障不改写实验失败或成功，也不作为策略性能结果。


## A完成后B启动资源门槛

A实际exit0；B尚无目录/模型调用时，GPU5 free43,754MiB未达原44,000MiB启动门槛，原supervisor按守卫退出。保留原EXITreason。后续实测GPU3/5 free49,813/64,919MiB恢复资源，只启动未开始的B，禁止重复A。恢复入口保留原pair/存储基线/写线，B还将资源等待计入2h额度，截止19:59:37。新状态文件bootstrap_card_pair_v1_recovery_receipt/EXIT避免覆盖原故障证据；原pair20:29与项目总24h截止仍保留。详见10_bootstrap_A_results.md。
