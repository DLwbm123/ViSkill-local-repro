# E：官方代码组件诊断 original_v1

2026-10-24，GPU4，固定base当前policy、本地Transformers BF16/SDPA、greedy、seed42。30分钟独立硬超时；原始开始1792781194.464318，截止1792782994.464318。任何同一尝试修复沿用该时钟，不重置；新算法修复需另立配置/记录并保留original_v1证据。

11项pytest断言：9通过、2失败，pytest退出1（40.95秒）。所有原始产物在`artifacts/component_tests/original_v1/`，JUnit为`junit.xml`。这不是E全通过，也没有PPO。

| 检查 | 结果 |
|---|---|
| FrozenLake真实图像与solver fixture动作成功执行 | PASS |
| 10维描述子及手算相似度 | PASS |
| quality与已有高分skill gate | PASS |
| 同batch不同seed、相同几何拒绝重复 | FAIL：应入1个，实际2个 |
| 轨迹帧顺序、路径标注、视觉卡、strategy可见/token成本 | PASS（图已目视） |
| outcome/format/prefix基础reward拆分 | PASS（仅官方代码口径） |
| paper状态进度反例 | FAIL：官方0、论文.75，总reward .02 vs .62 |
| 包含图像、strategy、geometry、trajectory、utility/use_count的库读写 | PASS |
| 相似检索与不相似阈值 | PASS |
| utility成功/失败EMA与使用计数 | PASS：.5→.55→.495，count2 |
| 当前本地policy strategy标签解析 | PASS（仅格式/非空检查） |

fixture为人工固定4×4地图`SFFF/FHHF/FFFF/FFFG`，官方BFS得down×3,right×3；7帧，仅组件测试，绝不写入训练库或cold-start。发布模型评价库也未修改。

本地policy使用与官方相同distill prompt、独立窗口及强制`<strategy>`前缀，最多256 token；实际生成26 token，prompt475 token。最终card918×192、image_grid1×14×66，对应231视觉token。初始帧、7个原始/标注帧、完整卡、原始策略、检索/重复/reward/utility证据均保存。目视确认轨迹依次向下3格再向右3格、动作标签和红色标注一致，strategy可读。

语义限制：生成strategy说“前方有洞时向下避让，然后一直向右”，未明确需要下移3格，对这个fixture的初始局部状态描述也不足够准确。因此只判定本地self-distillation调用与格式解析通过，不能称策略语义/完整E验收通过；不筛选重生成的有利结果。

后继修复路线应显式区分：保留官方prefix和批量gate原始路径；新增paper状态进度reward与插入前重新检查novelty的配置开关，补充正/负fixture。状态进度仍须实现完整Eq6中的额外步数惩罚，不能将本反例的l/Lz简化推广到所有轨迹。D仍待B/C；F必须等待D/E全通过及配对checkpoint验证准备。

## 显式修复与一次卡片复用诊断

`tests/test_audit_repairs.py`四项CPU回归通过（exit0、0.27秒）；未重新采样strategy。验证相对状态进度、额外步数惩罚、显式上界、缺失positions必须报错、同batch重复拒绝/新颖候选接收、positions序列化。原始9通过/2失败结果不改写。修复定义和非paper-exact边界见`docs/05_explicit_component_repairs.md`，还未证明训练agent集成和checkpoint配对。

固定一次card_transfer_v1，用相同base policy读取自己原先生成的测试卡，在同一solver fixture上greedy执行。结果失败：输出Down,Right，在(1,1)落洞；并非执行程序崩溃（EXIT0）。保存result.json，未重试、未挑选成功样本。该结果确认“strategy可解析/卡片可渲染”不足以保证语义可用，E整体仍PARTIAL/FAIL，不能解锁PPO。

下一候选修复须在运行前声明：self-distill提示增加已执行primitive action顺序/计数作为事实约束，仍由当前本地policy生成WHEN/WHAT，保留视觉轨迹卡；属于可见的prompt适配，不能称原始prompt复现。仅固定synthetic fixture验证，不使用D的held-out20种子调参；不得悄悄改成手写strategy或solver初始化训练库。

## 2026-10-24: action grounding remains insufficient

The original base-policy strategy omitted the repeated actions needed by the
six-action synthetic fixture. The predeclared action-grounding prompt appends
actual executed actions but leaves WHEN/WHAT generation to the local policy.

| Attempt | Render | Distill prompt tokens | Reuse prompt tokens | Result |
|---|---|---:|---:|---|
| grounded_fixture_v1 | default 256px, no path; configuration mismatch |1081|1191|Down,Right; hole|
| grounded_fixture_v2_render_corrected | original 120px, path enabled |542|651|Down,Right; hole|

Each generated once, seed42, greedy; neither was trained. Both EXIT0 indicate
completed diagnostics, not successful skill use. The corrected attempt took6.66s,
peak GPU allocated7.60GB. Its strategy still invents a hole ahead and says to turn
right then continue down, contradicting the observed down×3/right×3 route. Card
text is legible and trajectory order is intact; wrong semantics cannot be claimed
as a rendering or parser success. E remains PARTIAL/FAIL, F NOT RUN.

Next bounded diagnostic `policy_cycle_v1` uses fixed training seeds1–4 (no validation
seeds), base policy, original unmodified distillation prompt, an initially empty
isolated library, and only actual successful policy trajectories satisfying the
original quality gate. All four seeds are collected before admission; all four are
revisited once if any skill is admitted. This tests the mainline online data flow,
not generalization or optimization. All captions require review. No solver supplies
actions, and the diagnostic library will never initialize PPO.

`policy_cycle_v1` completed EXIT0 in22.94s: zero successes out of fixed seeds1–4,
zero eligible/admitted skills, so reuse correctly did not run. This is not a closed
loop PASS. No extra seed was selected to obtain a successful case.

A separately declared `policy_cycle_v2_train_sampling` retains exactly seeds1–4,
uses the official training sampling defaults (temperature1, top_p1, unlimited
 top_k, n1) instead of D's greedy evaluation settings, including self-distillation.
Transformers top_k0 corresponds to unlimited sampling. This tests a different
necessary training behavior, not a repeat until success. It still has no gradients,
no fixture actions, no cold-start, no test-set adaptation, and the original deadline.

## 2026-10-24 04:14: training sampling and persistence results

The fixed `policy_cycle_v2_train_sampling` diagnostic also finished EXIT0 with
0/4 successes, 0 eligible/admitted skills and no reuse episodes. Runtime57.19s,
peak allocated GPU7.73GB. It does not establish online self-distillation. Together
with the greedy result it shows that the proposed tiny batch of these four seeds
cannot yet demonstrate a library update; it does not establish that all training
seeds or the method cannot work. Do not scan seeds to choose a favorable smoke case.

Separately, the checkpoint sidecar implementation passed6 CPU tests (initial5 also
retained). It preserves skill images, trajectory positions, utility/use counts and
pre-restore backups; missing images, model-role directories, mismatched steps and
unsafe path overlap are rejected. Dummy role files were used. Real policy/critic
parameter save/load and training remain NOT RUN; E stays PARTIAL.

## Native SGLang diagnostic and bounded stop

`sglang_fixture_v1` reproduced the strategy problem under the native rollout
backend: its generated caption exactly copied the first prompt example, asserting
an adjacent goal although the fixture goal is three rows/columns away. Five turns
of Right did not finish. EXIT0 reports successful execution only, not E acceptance.

A final single-factor `sglang_fixture_v2_noexamples` removed only the example block
from the original distillation prompt (default include_examples remains true).
The output no longer copied the example verbatim, but still hallucinated a hole in
front and lost action counts. It executed Down,Left then repeated Down, failing at
five turns. Thus prompt-example copying is observable but does not fully explain
poor grounding. No hand-written strategies, added seeds, training, or resampling.
No further fixture prompt variants will be run in this preparation attempt.

New trainer-level persistence tests call the real `_save_checkpoint` and
`_load_checkpoint` methods with CPU dummy actor/critic workers. All3 passed: dummy
weights and skill utility restore together; missing images fail before either
worker loads; a failed sidecar save does not advance the latest-step tracker.
This is stronger integration evidence than file-only tests, but is still not a real
PPO model checkpoint or optimizer recovery. E semantic gate remains unmet; F is
NOT RUN. Mechanical repair success must not be used to erase these negative results.

## Released-policy capability contrast (2026-10-24 08:14)

Protocol: `configs/smoke/released_distillation_contrast_v1.json`, frozen before launch. Historical control is `sglang_fixture_v1`; only the pinned model/associated processor changed to r0. The original prompt/examples, fixture, render, sampling and action limits remain fixed. No released skill library was loaded, no base rerun and no prompt tuning. This distinct diagnostic retained the original overall start and 24-hour ceiling; the earlier six-hour preparation window was not reopened.

EXIT0 in 105.52 seconds including initialization; strategy/reuse section 10.46 seconds. One strategy generation: 475 prompt tokens, 20 completion tokens. It copied the same adjacent-goal example verbatim as the base model, although the fixture goal is at (3,3) and player at (0,0). Strategy parseability PASS, semantic grounding FAIL. The card was visually inspected: all seven frames and the down3/right3 path are intact and the incorrect caption is legible.

Reuse succeeded in two turns. Both responses proposed Down×3, Right×3; max_actions_per_step=5 executed Down×3, Right×2 on turn1. Turn2 executed three boundary Down actions and one Right before success. Thus nine executed actions, rewards 0.02 then1.02, retrieval score0.95. This demonstrates successful environment interaction with this generated card, but does not establish that the incorrect caption caused or helped success. No caption/trajectory ablation was performed.

The release checkpoint can complete this card-guided fixture while still failing self-distillation semantics. Base weakness alone therefore does not explain the observed caption error, and D-C success cannot stand in for E strategy acceptance. Neither base nor release is claimed to have passed all E gates; F remains NOT RUN. No new prompt, seed or checkpoint search follows this one-shot diagnostic.

Artifacts: `artifacts/sglang_fixture_released_contrast_v1/`; receipt and EXIT: `docs/evidence/components/released_contrast_v1/`. Peak four-root storage35,777,069,056bytes (increase258,048bytes); conservative total35,777,077,248bytes including existing8KiB cache. All project processes exited, GPU4 released, other jobs untouched. The previously recorded nonfatal CUTLASS CACHE_FILE atexit warning recurred.
