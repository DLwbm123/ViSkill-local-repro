# ViSkill 论文—代码审计（阶段 A）

审计日期：2026-10-24（Asia/Shanghai）。范围：完整阅读所提供的 23 页 v1 PDF（正文、Algorithm 1、附录 A–D、全部 prompt templates），审计 FrozenLake 的实际训练及评估调用链。未安装训练环境、未运行 VLM、未启动 PPO。本文的“代码存在”不代表动态验证通过。

## 1. 来源、范围和资源约束

- 论文：`2610.12403v1`，标题 *ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills*。输入 PDF 的路径在 `reproduction_manifest.json`；逐页文本见 [paper_fulltext.txt](evidence/paper_fulltext.txt)。表 5/6 已对照 [PDF 第 17 页](evidence/paper_pages/page17.png) 目视复核。
- 官方仓库：<https://github.com/ZJU-REAL/ViSkill>，固定 commit `7acaff880871b8d898d91c515c151080eabcba20`。
- 工作目录为本文件所在仓库，分支 `repro/local-no-api`；本轮仅增加审计材料，不改官方核心实现。
- 当前用户资源上限：**pro5000 最多 3 张 GPU**。不是已分配 3 张卡，不代表型号、单卡显存或可用容量已经查证。
- 本轮仅阶段 A。后续必须按 D/E → F 的顺序，D/E 全部通过才允许 PPO smoke；G 仅准备脚本，不自动运行 reduced/full training。
- 冻结要求：FrozenLake 优先、cold start=false、在线蒸馏=self、禁止外部推理服务/付费 API、禁止上传。后续低资源配置须与 paper 分离。

## 2. 算法与数据流

论文每个任务先提取初始几何配置，检索最高分技能，再将技能图像加入 VLM 的视觉上下文。策略输出 `<think>…</think><answer>…</answer>`，环境执行最多 5 个原子动作/turn，最多 5 turns。成功轨迹只有满足长度质量门与新颖性门才可由当前 policy 自蒸馏。库更新与 PPO 参数更新共同构成主线；solver 仅用于质量预算或独立组件测试 fixture。

核心数学定义：

1. Eq.1：技能 `z=(C,g,u,n)`，`C=Render(d,Annotate(trajectory))`，另存参考动作轨迹。
2. Eq.2：`q=0.9*sim+0.1*u`，最高分达到 `δr=0.8` 才检索；FrozenLake Eq.9 的方向、距离、hole 三个块在代码中权重为 `0.25/0.25/0.5`。
3. Eq.3：`u←0.9*u+0.1*y`，使用次数 `n←n+1`；新技能 `u=0.5,n=0`。
4. Eq.4：仅成功且 `L≤BFS shortest length`、`qmax<0.9` 才入库。Eq.5：容量满后淘汰 `u*log(n+1)` 最小者。
5. Eq.6/7：失败且检索到技能时，`rskill=max(0,(2ℓ−L)/Lz)`；附录 A 定义 ℓ 为匹配到的最远相对状态进度。最终 reward 为环境累计奖励加 `0.8*rskill`，成功不加该项。
6. Eq.8/11：actor clipped PPO（0.2），critic clipped value loss（0.5），GAE γ=λ=1，仅 policy response tokens 参与损失，PPO epoch=1，无 entropy/KL 辅助项。

### 论文—代码完整映射

代码位置均相对于本仓库；行号对应上述固定 commit。**NR=NOT RUN**，指完整模块动态验收尚未执行；“纯函数探针”仅证明指定函数的输出，不是环境闭环。

| 论文模块 | 论文公式/算法步骤 | 代码文件 | 关键函数/类 | 配置字段 | 当前是否可运行 | 测试方法 |
|---|---|---|---|---|---|---|
| task sampler | Alg.1 L4、B.1 seeds | `vagen/gym_agent_dataset.py:101,172,216`; `vagen/main_ppo_skill.py:394` | `_generate_from_len_three`, `AgenticDataset`, `create_rl_sampler` | `data.train_files`, `data.seed`, YAML `seed/n_envs` | 实现存在；NR | 逐项导出 seed，检查 train/val 不交叉及可恢复 sampler |
| visual observation renderer | 3.1、B.1 RGB | `vagen/envs/frozenlake/frozenlake_env.py:116,253` | `FrozenLake.reset`, `_render_async` | `config.render_mode=vision`, `size=4,p=.8` | 存在；NR | 实际 reset，检查 PIL RGB 图像、player/hole/goal 与状态一致 |
| policy rollout | Alg.1 L7 | `vagen/agent_loop/gym_agent_loop_skill.py:57`; `verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py` | `GymAgentLoop.run`, server generation | `rollout.name=sglang,mode=async`, per-turn length | 存在；NR | 本地 VLM 真实图像→文本→step→episode 结束 |
| action parser | C.2 action format | `vagen/envs/frozenlake/utils/utils.py:7`; `frozenlake_env.py:141` | `parse_free_think`, `parse_response`, `FrozenLake.step` | `prompt_format=free_think`, `max_actions_per_step=5` | 存在；NR | 大小写、逗号、缺标签、非法动作、超长截断；统计实际执行动作 |
| trajectory storage | 3.1 τ、A reference | `vagen/agent_loop/gym_agent_loop_skill.py:215,338` | `AgentData`, `_collect_trajectory`, `extra_fields` | `trainer.rollout_data_dir` | 存在；NR | 检查 initial+每个原子动作 frame、actions、positions 对齐 |
| geometric feature extractor | Eq.9、A | `vagen/skills/features.py:103,114` | `FrozenLakeFeatureExtractor.extract` | env `desc`；10 维 | 存在；NR | 手工可计算地图，验证 sign/dist/6 hole bits |
| similarity function | Eq.9 | `vagen/skills/features.py:149` | `FrozenLakeFeatureExtractor.similarity` | 函数默认 `.25/.25/.5` | 存在；NR | exact match=1；按块逐项改变，核对加权相等率 |
| skill retrieval | Eq.2、Alg.1 L6 | `vagen/skills/library.py:35,45`; `provider.py:187`; agent loop L128 | `_score`, `retrieve`, `_inject_skill` | `retrieval.similarity_weight=.9,min_score=.8` | 存在；NR | 同时验证排名、阈值、空库、图像插入及 placeholder 顺序 |
| skill card annotation | Eq.1、A rendering | `vagen/skills/renderer.py:63,99` | `FrozenLakeSkillRenderer`, `render_frames_with_path` | `render.trajectory.draw_path/action_type` | 存在；NR | 逐帧路径与动作标注人工检查，原图单独保留 |
| strategy generation | Alg.1 L11、C.1 | `vagen/agent_loop/gym_agent_loop_skill.py:411` | `_self_distill`, `server_manager.generate` | `distill.mode=self`, ≤256 tokens | 存在；NR | 当前 rollout policy 生成 WHEN/WHAT，保存原文及解析结果，禁止外部请求 |
| strategy parsing | C.1 tags | `vagen/skills/api_distiller.py:10` | `parse_strategy` | `<strategy>` 强制前缀 | 存在；NR | 缺闭标签时的 fallback 需单独测；非空不代表策略合格 |
| skill card composition | Eq.1 | `vagen/skills/renderer.py:168` | `compose_skill_image` | strategy vision=true, thumbnail=120, scale=1 | 存在；NR | 打开图像，检查顺序、文字、尺寸、实际 processor token 数 |
| skill library serialization | A management | `vagen/skills/library.py:106,147` | `ViSkillLibrary.save/load` | `library.path` | 存在；NR | JSON+PNG 保存重载，核对 features/trajectory/strategy/u/n；缺图不得按成功算 |
| utility update | Eq.3、Alg.1 L9 | `vagen/skills/library.py:90`; `distill_callback.py:20` | `update_utility`, `DistillCallback.step` | `utility.ema_alpha=.1,initial=.5` | 存在；NR | .5→成功 .55；独立失败 .45；非被检索技能不变 |
| usage-count update | Eq.3 | `vagen/skills/library.py:94` | `update_utility` | `use_count` | 存在；NR | 每个已检索 episode +1；重复/validation 不额外计入 |
| quality gate | Eq.4、A solver | `vagen/skills/provider.py:233`; `solver.py:93`; `distill_callback.py:53` | `should_distill`, `FrozenLakeSolver.solve` | `distill.quality_ratio=1` | 存在；NR | 只接收成功且原子动作数≤BFS；不可解/零长度拒绝 |
| novelty gate | Eq.4、Alg.1 L10 | `vagen/skills/provider.py:233`; `distill_callback.py:53` | `should_distill`, `_filter_distill_candidates` | `distill.min_score=.9` | 存在；批内覆盖检查不足；NR | 同/近重复但不同 seed；同批候选逐项检验，不能只测同 seed 去重 |
| online library insertion | Alg.1 L11–12 | `vagen/skills/distill_callback.py:84`; `library.py:60` | `_build_skill_data`, `admit` | `library.max_size=10000` 脚本值 | 存在；NR | 空库质量成功→self strategy→完整 card→admit→次步检索 |
| capacity eviction | Eq.5 | `vagen/skills/library.py:160` | `_evict` | `library.max_size` | 存在；NR | 控制 u,n，验证最低分淘汰；log2 与 ln 的排序等价 |
| skill-guided reward | Eq.6/7、A progress | `vagen/skills/reward.py:6,20`; agent loop L269 | `compute_prefix_reward`, `compute_reward` | `reward.mode=skill_guided,guided_weight=.8` | **与论文状态进度不一致**；纯函数探针已执行 | 见 `evidence/reward_counterexample.json`，后续补状态重合但动作前缀不同的环境 fixture |
| outcome/format reward | A reward | `vagen/envs/frozenlake/frozenlake_env.py:141` | `FrozenLake.step`, `_is_success` | 环境 `format_reward=.02,success_reward=1` | 存在；NR | success=1 + 每有效 turn .02，失败及非法格式单独分解 |
| PPO actor | Eq.8、Alg.1 L15 | `verl/verl/workers/actor/dp_actor.py:372`; `vagen/ray_trainer_skill.py:329` | `update_policy`, actor `update_actor` | lr=1e−6, mini=32, clip=.2 | 存在；NR | 实际 loss/backward、finite grad、选定 trainable 参数 delta>0 |
| PPO critic | Eq.11、A GAE | `verl/verl/workers/critic/dp_critic.py:192`; trainer L323 | `update_critic`, `compute_value_loss` | lr=1e−5, cliprange_value=.5 | 存在；NR | value loss/backward、response mask、finite grad；不以 actor 成功代替 |
| checkpoint manager | PPO persistence | `vagen/ray_trainer.py:1166,1226`; `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py:98,180` | `_save_checkpoint`, `_load_checkpoint` | save_freq=100, resume_mode=auto | policy/optimizer 路径存在；**缺 paired skill snapshot**；NR | 保存两个不同库状态对应的 checkpoint，恢复较旧步验证库也回退 |
| validation rollout | 4.1、B.1 | `vagen/ray_trainer.py:814`; same skill agent | `_validate`, `GymAgentLoop.run` | `val_before_train=true,test_freq=20` | 训练内复用技能库；NR | 冻结库前后对比，验证 skill ID/card 一致，held-out 不入库 |
| cold-start initialization | 3.3、Alg.1 L1–3 | `vagen/ray_trainer_skill.py:60`; `vagen/skills/cold_start.py` | `_maybe_cold_start`, offline selection | `cold_start.enable=false` | 本阶段禁止启用；NOT RUN | 断言 false；solver fixture 与训练库路径隔离 |

## 3. 训练参数核对（正文、附录、代码分开）

以下是脚本覆盖 Hydra defaults 后的**静态推导**，尚未运行 Hydra 完整配置解析；运行前仍须保存 resolved config。裸 YAML defaults、Python fallback 与官方启动脚本不是同一配置。

| 参数 | Paper says / Appendix says（Table 5/6） | Code says（FrozenLake 官方脚本及继承） | 本次复现选择与理由 |
|---|---|---|---|
| backbone/算法 | Qwen2.5-VL-3B-Instruct / PPO+GAE | 一致 | 保留，后续不能改成 SFT |
| train batch / mini-batch / rollout n | 128 / 32 / 1 | 128 / 32 / 1 | paper 原样留档；smoke 待硬件审计确定合法整除关系 |
| actor / critic LR | 1e−6 / 1e−5 | 一致 | 保留，不在本轮调参 |
| weight decay | .01 | actor optim/fsdp 与 critic optim 默认 .01 | 一致，待 resolved config 复核 |
| GAE γ / λ | 1 / 1 | skill YAML 1 / 1 | 一致 |
| PPO epochs | 1 | actor 默认 1，critic 跟随 | 一致 |
| actor clip / critic clip | .2 / .5 | 默认 .2 / .5 | 一致 |
| entropy / KL | 0 / 0 | entropy=0, use_kl_loss=false, use_kl_in_reward=false, kl_ctrl.kl_coef=0 | 无 reference worker；不是一定同时加载三份完整模型 |
| rollout memory utilization | .6 | .6 | 一致；不是整体训练显存占用上限 |
| initial utility / α / β | .5 / .9 / .1 | skill YAML 一致 | 一致 |
| δr / δd / guidance λ | .8 / .9 / .8 | 脚本一致 | 一致；裸 YAML reward.mode=success，必须脚本覆盖 skill_guided |
| training GPUs | 正文 §4.1：4×H100；附录 Table 6：2 | `CUDA_VISIBLE_DEVICES=0,1,2,3`，4 卡 | 无法消除论文矛盾。保留两份原始证据；用户最多 pro5000 3 卡，后续资源缩减单列，不标为已完成 full reproduction |
| training steps | 300 | `trainer.total_training_steps=301`；fit 在达到 total 后结束 | 显式偏差 1 step；不静默修改，smoke 另外配置 |
| max prompt / response | 5000 / 4000 | 脚本一致；裸 skill YAML 为 9000 / 8000 | 采用脚本作为 paper baseline 留档 |
| per-turn response / turns / actions | B.1：5 turns、5 actions，C.1 self≤256 | task YAML response=512、5 turns、5 actions；self≤256 | 记录总 response 与每 turn 的限制不同 |
| map / stochasticity | B.1：p=.8，deterministic | size=4,p=.8,is_slippery=false,slip_prob=0 | 保留；不因 prompt 的 slippery 文本就改变 transition |
| seed split | B.1 disjoint train/eval | train 1–10000；validation 10001–10256，各不重复 | 保留，组件 solver fixture 不用于训练初始化 |
| format / terminal reward | A：.02 / 1 | `FrozenLakeEnvConfig` 一致；provider reward.format_reward=.1 未用于该环境 step | 使用实际环境字段，防止错改无效 provider 字段 |
| skill capacity | Nmax；Table 5 未给具体值 | library class fallback 32、skill YAML 100、脚本 10000 | 以具体启动配置为准；paper 未披露值不自行补造 |
| cold start | 可选；γ=.2 的变体，外部 GPT-5.4-mini | README 默认关闭；训练脚本 false；cold YAML size=20 | 本阶段 false；非 cold-start 变体是论文合法 setting |
| checkpoint / validation | Table 5/6 未列 | 每100步 checkpoint、每20步 validation、先验证、各保留1份 actor/critic | smoke 将来需 save_freq=1 与 paired library，不在 A 阶段启动 |

## 4. 核心不一致与影响

### A-01：reward 语义不一致（影响算法忠实性）

- **Paper says**：§3.3 Eq.6/7 基于 ℓ；**Appendix says**：A. reward 明确以相对初始状态的最远匹配进度定义 FrozenLake ℓ，而非动作对齐。
- **Code says**：`GymAgentLoop._compute_reward` 将动作字符串直接交给 `compute_prefix_reward`；第一处动作不同后停止前缀计数，没有使用收集到的 `trajectory_positions`。
- **实测证据**：无障碍抽象轨迹，reference `R,R,D,D`，executed `D,R,R`；执行轨迹达到 reference 的第3个位置且尚未到目标。论文公式 rskill=.75，官方函数=0；假设一个合法 turn，则总 reward 分别 .62 与 .02。见 [reward_counterexample.json](evidence/reward_counterexample.json)。这是纯函数反例，未执行环境 episode。
- **本次复现选择**：保留官方实现作为可追溯的 code-as-released 路线；后续在独立配置中实现并验证 paper-state-progress 路线，再决定 D/E/F 的使用。当前不称其为 faithful paper reward。
- **选择理由**：动作前缀与状态进度不等价，不能静默修改或仅改函数命名掩盖。

### A-02：技能库与 policy checkpoint 没有按步绑定（影响恢复）

- **Paper says**：库与 policy 共同演化；未具体规定 checkpoint 打包协议。
- **Code says**：每个 rollout batch 后 callback 更新全局 `skill_library/skill_index.json` 与 PNG；trainer `_save_checkpoint` 仅保存 actor/critic、optimizer/extra、dataloader 与 latest iteration。`_load_checkpoint` 不恢复技能库，provider 根据独立的 `library.path` 加载当前文件。
- **本次复现选择**：后续需要将技能库 snapshot 与 checkpoint step 一起保存并在恢复时绑定；本轮只审计，不改源码。
- **选择理由**：若 step101已更新库但最近 checkpoint 是step100，恢复后不能证明库与参数属于同一步。记录路径不等于可恢复快照。

### A-03：门控与库更新是批处理，和 Algorithm 1 顺序不同

- **Paper says**：Alg.1 L6–12 每条 episode 后依次更新效用、门控和入库，后续任务可看到新库。
- **Code says**：`fit` 先收集整个 batch，agent 在 rollout 中 self-distill；callback 一次更新所有效用，再根据 rollout 时的 `skill_score` 筛选全部候选，最后逐项入库。只按 seed 去重，没有以已新增候选重新计算几何覆盖。
- **影响**：不同 seed 的近重复地图可在同一 batch 同时进入空库；不是全库几何 novelty 不存在，而是缺少批内再检查。utility 更新相对于 novelty 的时点也不同。若检索图像失败或最高分低于阈值，`skill_score` 为 None，而非总是保留 qmax；默认 δr<δd 时阈值低的情况不改变 novelty 判定，但自定义阈值时不能假定等价。
- **本次复现选择**：保留实现行为和 negative case；E 阶段必须用“不同 seed 同几何/近重复”测试，不能只测 seen_seeds。
- **选择理由**：在证据不足时，不把批处理实现自动解释为 Algorithm 1 的完全等价实现。

### A-04：蒸馏失败未作为拒收门

- **Code says**：`parse_strategy` 缺闭标签时返回完整原文（连强制前缀也可能保留）；callback 允许 strategy 为空时构造并 admit 技能。图像加载缺失时 provider 可以退回 text-only。
- **本次复现选择**：E 阶段要求 strategy 原文、tag解析、WHEN/WHAT、图像存在性全部留证；不能把 fallback 当作完成 visual-native 机制。此轮未修复或声称失败已发生。

### A-05：论文内部 novelty 阈值解释方向矛盾

- **Paper says**：Eq.4 使用 `qmax<δd`。§4.4 却称 .85 过于宽松、.95 过于严格。
- **Code says**：`score>=distill.min_score` 拒绝，和 Eq.4 一致。固定 score 时，提高 δd 实际放宽入库条件。
- **本次复现选择**：数学不等式与真实代码为准，保留图5及文字的原始声明，不替作者推断曲线/标签是否颠倒。

## 5. 必答审计项（14项）

| 项目 | 审计结论与证据 |
|---|---|
| 1. FrozenLake GPU 一致性 | 不一致：正文4、附录2、脚本4。见上表与PDF17页截图。 |
| 2. Table5/6 与默认值 | 官方脚本多数一致；301 vs300、4 vs2；裸 YAML reward=success、长度9000/8000、容量100不等同官方脚本。 |
| 3. 全参/部分/PEFT | actor `lora_rank=0`、`freeze_vision_tower=false`；critic lora_rank=0，`fsdp_workers.py` 仅显式 freeze 时冻结视觉塔。静态指向全参 PPO（含 critic value head），不是默认 LoRA。实际 trainable parameter 清单仍待运行。 |
| 4. actor/critic/ref 部署 | Ray global_pool；FSDP actor+async SGLang rollout，tensor-parallel=1；单独 critic worker 同池；actor/critic 参数与优化器 CPU offload。KL两个开关为false，故不注册 reference role。若开启KL则新增同池 ref worker；路径名 REF_MODEL_PATH 本身不表示 reference 已启用。 |
| 5. strategy 使用当前 policy？ | 是：`_self_distill` 调相同 `server_manager.generate`，独立推理窗口、强制 strategy 前缀、≤256 tokens；发生在当前 batch PPO 更新之前。外部 api distiller 仅在 mode=api 时实例化，冷启动另走外部路径。 |
| 6. 库更新时机 | 每个训练 rollout batch 后、GAE/critic/actor 更新前 callback；不是每个episode立即全局共享，详见A-03。 |
| 7. 库与checkpoint同步？ | 否，无按step snapshot/reload，见A-02。单独调用 save 不等于 synchronized checkpoint。 |
| 8. validation 使用训练库？ | 训练内 `_validate` 使用同一个 skill agent / library.path，provider 每次检索重新读磁盘。验证输出不进入 training callback；验证成功仍可能执行 self-distill 推理但不入库。未动态验证。独立 `examples/evaluate` 的 generic workflow 不接入技能库。 |
| 9. release 包含完整库？ | 见下方“发布资产核查”；远端清单/metadata 与本地下载、加载验证分开，绝不由模型名称推断完整性。 |
| 10. 默认外部 API？ | FrozenLake evaluation 默认 backend=openai、model=gpt-4.1-mini、base_url=null，client可读取环境OPENAI_API_KEY，绝不能原样运行；cold YAML外部模型；训练 self/cold=false 无需外部模型。 |
| 11. telemetry / uploads？ | 训练脚本 `WANDB_MODE=offline` 但 logger 包括 wandb；后续需 console-only/WANDB_MODE=disabled。HF manager 有 create_repo/upload_folder，`hf_save_freq=null` 时未激活；若误设频率而 private=false，可能创建公开库。依赖 Ray/HF/wandb telemetry 应在C阶段显式关闭并检查联网；本轮不能证明依赖无遥测。 |
| 12. 依赖未固定？ | `setup.py` gym-sokoban/gymnasium 未固定、uvicorn仅上界；install脚本 transformers>=4.51、torch间接解析、numpy<2、ray/peft/qwen-vl-utils/wandb/opencv等未锁。固定的部分也存在互相冲突，见下节。 |
| 13. bundled veRL差异 | 版本声明0.6.1；已从公开上游v0.6.1标签比较真实文件，5个源码文件有修改，详见下节；不是仅凭version判同版。 |
| 14. README/论文/脚本差异 | GPU/steps、安装Python与wheel、评估不带库、service绑定0.0.0.0、独立评估忽略per-turn字段等，均列于下节。 |

## 6. 安装与评估入口风险（未安装、未启动）

1. README 推荐 Python3.10，但 `verl/scripts/install_vllm_sglang_mcore.sh` 下载 FlashAttention wheel 硬编码 **cp312/Linux/torch2.8/cu12**；与3.10不兼容。不能原样执行后把最终 echo 当成功。脚本本身无 `set -e`，中途 pip/wget 失败可能继续。外层 install.sh 的 `set -e` 不会自动让子脚本中每条命令失败即停。
2. 安装脚本固定 `sglang[all]==0.5.2`，但 bundled setup extra sglang=0.5.5/torch2.8；顶层 `trl==0.26.2`，bundled extra trl<=0.9.6；脚本 pyarrow>=15 与 setup>=19，存在版本约束来源差异。暂未选定环境 lock，不把可安装性当已证实。
3. `install_render.sh` 会安装 ai2thor、下载 navigation scenes、安装 ManiSkill并下载资源，当前阶段不运行。FrozenLake 仍需要 pygame/PIL 等真实渲染依赖，应最小安装。
4. FlashAttention 对纯 Transformers evaluation 可研究 SDPA，但当前 bundled PPO valuehead loader 明写 `attn_implementation=flash_attention_2`；不能一概视为可选加速并直接绕过。PPO fallback 必须检查具体路径并保留改动记录。
5. actor/critic FSDP 的 `engine/fsdp.yaml:model_dtype=fp32`，actor loader在未指定时也使用fp32，critic同样有fp32默认。混合精度计算和rollout BF16不意味着全部模型/优化器存储都是BF16；B阶段必须分项估算。critic patch 在某些输出路径会构造/升精度巨大的 logits，这是额外显存风险，尚未动态触发。
6. 独立 evaluation `GenericVisionInferenceWorkflow` 不调用 `ViSkillProvider`，仅修改 checkpoint/model 不能得到“checkpoint + library”的 C 组。需要独立 wrapper 接入正式技能检索，或使用训练内 val-only 并核查代价。
7. 独立 eval YAML 写 `response_length_per_turn=512`，但 `_parse_env_specs` 不把它传入 `EnvSpec`；不能认定512已生效，应走 chat_config 并核查请求参数。
8. `examples/evaluate/launch_sglang.sh` 默认 `--host 0.0.0.0`，不是严格loopback；后续本地服务须绑127.0.0.1，显式校验 endpoint host，拒绝重定向外部请求，dummy key 不读取真实 key。优先直接 Transformers。
9. 训练脚本 `set -x` 及入口/参数包含项目、方法信息，不满足用户可见进程命令行规则。后续必须另写中性启动入口与参数传递、检查子进程，不运行原脚本规避该要求。

## 7. bundled veRL 与上游的实证比较

公开对照：`volcengine/verl` 的 **v0.6.1**，commit `d62da4950573d7a4b7ef2362337952e7ab59e78d`。两个官方公开仓库均以显式Git HTTPS代理克隆；未调用模型API。

完整文件级比较结果见 [verl_tree_comparison.json](evidence/verl_tree_comparison.json)，核心包差异见 [verl_package_diff.patch](evidence/verl_package_diff.patch)。5个代码文件修改、81行新增/3行删除；另有上游的 `.github`、`.gitignore`、`.vscode` 等未随bundled树带入。对照标签不是作者声明的确切fork base，不把标签推断写成git历史事实。

| 文件 | 实际修改 | 与复现的关系 |
|---|---|---|
| `verl/models/transformers/monkey_patch.py` | 添加 VLM valuehead forward，处理 logits缺失、hidden_states、value输出并将logits转fp32 | critic兼容性及显存风险，不可直接用pip安装上游替代 |
| `verl/utils/model.py` | 将 text_config.hidden_size 补到顶层 | TRL valuehead初始化适配 |
| `verl/workers/critic/dp_critic.py` | `response_mask *= value_mask` | critic可忽略额外token；本脚本标准GAE不使用no_concat专门value_mask |
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | 将传入max_new_tokens纳入最小上限 | 保证per-turn及self-distill上限 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 同类token限制修复 | 未来切换后端需保留 |

## 8. 发布资产核查

本节结果以 `docs/evidence/released_model_manifest.json` 与（若存在）`released_skill_index.json` 为准。仅下载公开模型文件清单与轻量元数据，不下载大权重；不运行模型，不将远端metadata检查视为 D 阶段 PASS。

后续 D 阶段必须同时确认模型权重、processor/tokenizer、generation config、skill_index、PNG、10维特征、参考动作、strategy、utility、use_count，并逐项检查引用路径能在下载目录解析。已核实公开模型 revision **`edafb2b71f73afb1c35b82d264346c2ae2b5981e`**，文件清单共284项：4个safetensors权重分片、权重索引、config/generation_config、tokenizer/processor配置，以及 `skill_library/skill_index.json` 与 **265张PNG**。索引包含265个技能；所有条目具备id、features、trajectory、strategy、image_path、utility、use_count；10维特征检查与远端PNG引用检查通过；空strategy=0，空trajectory=0。证据见 [released_asset_checks.json](evidence/released_asset_checks.json)。

结论：**远端发布清单与技能元数据包含复现所需结构**。本地尚未获得大权重/PNG、尚未加载模型或库，不宣称发布checkpoint实际rollout成功，也不确认发布库与某一policy训练step的历史绑定。

## 9. 后续验收边界与唯一优先动作

本轮确认代码组件大体齐备，但尚未验证真实 visual observation→VLM→环境→在线技能→PPO 参数更新闭环；不标记 `ViSkill preparation completed`。审计发现的核心偏差不得通过删模块或静默改算法处理。

**下一步唯一优先动作：对 pro5000 在最多3卡约束下执行阶段 B 硬件与资源审计**，先读服务器与实验约定，核查实际可用卡、显存、RAM/磁盘及既有负载，结合FP32参数/优化器、BF16计算、序列长度、rollout KV cache、critic valuehead推导可行配置。此后才进入最小环境与D/E验证。

本次没有模型推理结果、成功率、PPO loss、parameter delta或checkpoint重载结果。所有计划测试方法均未冒充已经执行。
