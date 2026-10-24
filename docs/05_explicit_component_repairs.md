# 显式组件修复方案 v1

目的：让FrozenLake主线在不隐藏官方实现差异的前提下满足视觉状态奖励与novelty要求。原始实验`original_v1`已退出1，9通过/2失败及原始strategy均冻结。以下是独立、预先声明的工程修复配置，不追改原始结果，不调整20个held-out评价种子的提示词、模型或超参数。

| 位置 | 官方/原值 | 修复配置 | 含义与忠实性 |
|---|---|---|---|
| reward mode | skill_guided动作前缀 | state_progress_bounded | 使用真实逐动作位置相对初始点匹配最远参考点，完整保留Eq6额外步数惩罚；默认旧模式不变 |
| reward上界 | Eq6仅max(0,...)，正文声称≤1 | 显式min(1,...) | **非paper-exact的边界修复**：参考down,right,up,right，在不同地图以right,right到参考末端，原式(2×4−2)/4=1.5；上界1保持λ=.8小于成功奖励1的设计意图。原式也以bounded=False提供，不隐瞒差异 |
| batch novelty | rollout时score，候选批量筛选 | recheck_batch_novelty=true | 插入前按当前内存库重新计算composite score≥.9则拒绝；避免provider.retrieve重新读盘丢掉批内新增。False仍为官方默认 |
| skill positions | 库只存动作 | 新生成FrozenLake技能额外保存positions | 实际轨迹而非假设动作一定移动；保存/加载一致。新reward遇缺失position报错，禁止静默退回prefix或从动作伪造state |
| invalid turn记录 | 无动作仍追加cached位置 | 只有实际动作才可能追加位置 | 避免positions与actions长度不一致；FrozenLake正常动作使用环境逐动作step_positions |

范围仅FrozenLake。B/C仍在原始远程源码、官方prefix reward口径下执行；修复代码待这两臂退出后才部署。不会对发布库补造positions，也不会改发布模型来追分。CPU回归只用固定synthetic地图与保存的本地strategy，无新增模型采样、无训练；原始失败测试保留，修复测试另存`tests/test_audit_repairs.py`。

尚未解决：strategy语义质量只能按实际输出评估，当前可解析不等于完整语义验收；F的policy/skill checkpoint配对和PPO运行依赖仍待实现。上述机械修复通过也不能单独解锁F。

四项CPU回归已PASS，结果在artifacts/component_tests/repairs_v1；原始整套失败和strategy transfer失败仍保留。新增上界反例是明确公式/正文边界矛盾，修复路线标记非paper-exact。

## Action-grounded self-distillation diagnostic (not adopted as smoke default)

The original image-based prompt omits the text action sequence. An opt-in
`skill_system.distill.ground_actions=true` now appends observed primitive actions
and asks the same local policy to retain direction order/counts. The default is
false, and the API path is untouched. This changes the distillation prompt and is
not paper-exact. No strategy text is supplied by the implementation.

One fixed solver fixture (`SFFF/FHHF/FFFF/FFFG`) was tested greedily. Attempt v1
inadvertently used default 256px/no-path rendering, unlike the original component
fixture's 120px/path rendering. This configuration error is retained, not used as
an isolated prompt comparison. Attempt v2 explicitly restores the original render
configuration. Both generated semantically incorrect strategies and executed
Down,Right into a hole. The option is not promoted to the proposed PPO configuration.
See `configs/smoke/grounded_fixture_v*.json` and corresponding component artifacts.
No test library is a training cold-start library.

## Checkpoint pairing (opt-in local persistence repair)

`skill_system.checkpoint_pairing.enable=true` adds a sidecar through three trainer
hooks. The original default is unchanged. After actor/critic/dataloader saves, the
trainer snapshots the active library index and referenced visual cards into
`global_step_N/paired_skill_library`, then publishes `skill_pair.json`, before the
latest-step tracker is written. Empty libraries remain explicitly empty; fixtures
are never inserted to make a checkpoint pass.

On resume, the step, role directories, count, unique IDs and referenced image paths
must validate before model loading. After model/dataloader loads, the snapshot is
restored to the active library path and the callback's in-memory library is reloaded.
The previous active directory is kept as `*.before_restore_*`, with rollback if the
replacement fails. Active/checkpoint path overlap is rejected. Missing sidecars fail
closed rather than combining current memory with an older policy. Remote HDFS is
unsupported in this opt-in path. These checks do not prove bytewise model identity.

`tests/test_checkpoint_pair.py`: five initial CPU checks passed, then six passed
after adding ancestor-path rejection. Model role files are explicitly dummy test
files. This proves sidecar persistence and rejection behavior only; actual full-param
PPO save/load and numerical weight comparison remain NOT RUN. The proposed small
checkpoint excludes optimizer state, so even a future successful weight/library
restore must not be described as optimizer-continuous resume.

The three trainer-hook integration tests also pass using CPU dummy workers and
actual trainer save/load methods. They verify sidecar failure ordering, unchanged
latest-step tracker on failed save, and dummy actor/critic plus library restoration.
They do not load a real Qwen actor/critic checkpoint.

`distill.include_examples=false` is a separate opt-in prompt diagnostic. The native
backend exactly copied one original example, motivating a single-factor removal.
The variant still failed semantic grounding and task reuse. Default remains true;
this unsuccessful variant is not adopted for PPO. No further fixture prompt
variants are scheduled in this preparation attempt.
