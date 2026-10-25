# ViSkill local reproduction and audit

Research handoff: operational updates and bounded inference recovery are verified, but stable parameter-learning benefit remains unestablished. Closed protocols retain their negative results and engineering failures; no GPU experiment awaits continuation. [Report31](reports/31_research_handoff.md) maps the evidence, unknown causes and prerequisites for a different next-stage study.

Dynamic-retrieval CPU screen: all40 saved episodes/132 actual model-turn starts yielded zero new valid-card opportunities among initial non-hits, including0/8 long episodes. The preregistered feasibility gate failed; no GPU, model calls or environment steps. See [report30](reports/30_turn_retrieval_cpu_screen.md). This is a posthoc opportunity screen, not efficacy evidence.

Curriculum pair closed after collection: all40 episodes/134 calls retained, but two unintended distillation calls violated the frozen zero-distillation contract. No actor update or Dev evaluation occurred; long-start success0/8 also misses the prespecified positive-feedback prerequisite. See [report29](reports/29_curriculum_collection_failure.md). This is an engineering/protocol failure, with no paired learning result.

CPU input readiness: the fixed gzip1 container reduced one existing C32 input from290.59MB to27.54MB with basic loading checks, zero model operations. Private inputs stay remote. A new support-matched curriculum comparison now has a frozen80-board protocol and CPU-checked interfaces; no model calls or results at publication. See [report27](reports/27_input_container_feasibility.md) and [protocol28](reports/28_support_matched_curriculum.md).

CPU retention screen:9/16 natural Dev starts have BFS4–6 while C32 starts stop at3; this does not establish the cause of the single natural regression. Tiny AdamW fixtures retain an epsilon-sensitive counterexample, and identical-reference KL/EWC have zero initial penalty gradients. No new GPU experiment; a support-matched curriculum remains design-only with input-storage readiness unresolved. See [report26](reports/26_retention_cpu_screen.md).

Answer-region loss numerator pilot: conditional success/quality rose from original b0 4/16 to A 7/16, but other-natural success/quality fell2/16 to1/16 and holes increased5 to6. The frozen investment gate failed; this protocol closes without extra mask/LR/seed search. All96 episodes/367 calls and one new actor step are reported in [report25](reports/25_answer_region_step.md). Paired intervals include zero; no stable learning benefit is established.

Bounded inference-artifact recovery: existing Z loaded from its full snapshot and from retained b0 plus a private BF16 delta produced identical logprobs on two fixed saved inputs (371 generated tokens). Eight actual actor forwards, zero training/generation; the original zero-score initialization fault remains recorded. This is operational recovery evidence only; see [report24](reports/24_bf16_delta_reload.md).

Saved-batch CPU audit: answer-overlap tokens account for5.11%/3.52% of absolute scalar logprob-loss coefficient mass in O/Z; this is **not parameter-gradient attribution or efficacy evidence**. All32 rows admit a nonempty answer mask; no new model forward or training. See [report22](reports/22_actor_loss_support_audit.md).

Prospective validation (2026-10-25): the zero-head performance hint **failed its frozen gate** on32 new boards. Conditional success/quality were7/6 for original b0,3/3 for O,4/4 for Z (each/16); Z had6 success declines against b0. Other-multistep Stop turns increased4 to10. All96 episodes and364 calls completed with0 new training. This closes the zero-head-only performance route; see [report21](reports/21_prospective_value_head_eval.md). Historical results and the retained zero-query initialization fault are preserved.

Latest single-step control (2026-10-25): the zero value-head arm gained **one conditional success (0/8 to1/8)**; the original-head arm stayed0/8 and both other-multistep panels stayed1/8. Paired intervals include zero. The two historical Stop transfer probes failed the predicted direction, and the fresh training batch contained no Stop turns. No stable learning benefit or collapse cause is established. All96 episodes,338 native calls, retained pre-update serialization failure and numeric signals are reported in [report20](reports/20_value_head_single_step.md). Two inference snapshots were retained privately at that stage; subsequent independent interventions are reported separately.

Latest kernel reference (2026-10-25): disabling fused output kernels on the same8 actor layouts left **all8 probability comparisons outside the frozen tolerance**, with errors close to the fused result. Shared causal input prefixes match exactly in tokens, positions and image tensors. No new native requests, generated tokens, environment actions or optimizer steps. This kernel branch is closed; no learning benefit or collapse cause is established. See [report19](reports/19_kernel_score_reference.md).

Latest diagnostic (2026-10-25): **all8 saved-request comparisons failed the frozen all-token probability-error gate**. Differences concentrate in think tokens; the8 observed action tokens differ by at most.001381 nats. Reconstructed first-batch signals match seven historical aggregate metrics. Two Stop action tokens have positive original whitened advantages and negative zero-baseline counterfactuals; no causal update test has been performed. Zero-generation and dispatch-contract incidents remain failures (9 dispatch attempts,8 actual prefills,1 unwanted unused newline), with0 environment actions or optimizer updates. See [report18](reports/18_recorded_backend_audit.md) and [numeric evidence](docs/evidence/components/recorded_backend_audit_v2/summary.json).

Independent local reproduction of **ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills**, using local inference with no external model API. This is not the official repository.

**Latest result: fixed-library PPO failed its preregistered learning gate.** Both arms completed four training batches and all before/midpoint/final evaluations: 1,408 complete episodes and 5,135 model requests. Card exposure and real parameter updates were verified; performance deteriorated on the primary conditional task.

| Same fixed cards, conditional Dev32 | B before training | A after ordinary PPO | B after card-conditioned PPO |
|---|---:|---:|---:|
| Success |11/32|9/32|1/32|
| Shortest-path quality eligibility |10/32|9/32|1/32|

B versus its own baseline had zero improved and ten regressed success cases. B versus A had one improvement and nine regressions. All four primary success/quality conditions failed. Natural multi-step success with cards stayed at6/53, but quality fell from6/53 to4/53; unchanged aggregate success does not mean that the original successful cases were preserved. This protocol is closed without additional PPO batches, seeds or skill-reward training.

## Review these first

- [Report16: full results, paired comparisons, uncertainty and gate decision](reports/16_fixed_library_results.md)
- [Report17: failure behavior, numerical evidence and unresolved causes](reports/17_fixed_library_failure_audit.md)
- [Frozen fixed-library PPO protocol](reports/15_fixed_library_ppo.md)
- [All1,408 episode outcomes](docs/evidence/components/fixed_library_ppo_v1/result.json)
- [Read-only failure audit](docs/evidence/components/fixed_library_ppo_v1/failure_audit.json)
- [Four-condition visual-content controls](reports/14_card_content_controls.md)
- [Complete experiment summary](reports/06_results_summary.md)
- [Chronological status](reports/REPRODUCTION_STATUS.md)

The fixed-library experiment deliberately used five cards produced by the earlier online bootstrap and a conditional curriculum. Skill-reward weight remained0; there was no online library admission or utility update. This is an explicit mechanism variant, not the paper's full300-step training. Final weights were not retained, so independent restoration or retrospective evaluation is unavailable. Original free-text E and full F remain PARTIAL. A proposed eight-request scoring diagnostic has not started.

## Earlier results retained

The earlier online bootstrap completed576 episodes/2,056 calls. Its B arm went from9/64 to10/64 successes, but multi-step success fell from6/53 to2/53. A learning benefit was not established. It produced four one-step cards and one three-step card; that three-step card had no later training exposure. See [report11](reports/11_bootstrap_pair_results.md).

Independent frozen-weight controls found an immediate context effect on a narrow Up-three-times applicability set: no card/real card/blank card/text yielded0/9/1/4 successes, each out of16. This justified the later bounded learning test; it does not override that test's negative result. See [reports12-14](reports/12_multi_card_transfer.md).

The initial128-episode capacity run produced15 successes,8 quality-eligible trajectories,5 valid strategies and3 admitted one-step skills. All three eligible multi-step trajectories failed distillation. Prompt/schema/observation and dynamics-wording diagnostics did not meet their improvement gates. Earlier one-batch PPO and model reload proved real updates and empty-library restoration; these engineering results do not establish scientific learning gains.

## Implementation and evidence boundaries

The named directional variant changes strategy representation and uses privileged deterministic state for reject-only semantic/admission/retrieval checks. Original free-text failures and paper/code discrepancies are retained. Current-policy self-distillation and disabled cold start do not require an external model API. Actor self-consistency, finite gradients and weight-sync receipts do not yet establish SGLang-versus-production-actor probability equivalence. No claim about the paper's full-scale outcome follows from these reduced experiments.

## Source and public scope

- [Official repository](https://github.com/ZJU-REAL/ViSkill), pinned upstream7acaff880871b8d898d91c515c151080eabcba20.
- [Paper](https://arxiv.org/abs/2610.12403), [paper/code audit](docs/01_paper_code_audit.md), [upstream README](README_UPSTREAM.md).
- [Environment deviations](docs/03_environment_setup.md), [publication manifest](PUBLIC_RELEASE.json).

This sanitized snapshot includes code, configs, synthetic map manifests/images, episode outcomes, audit scripts and compact numeric/update/sync/exit evidence. Model weights, checkpoints, complete runtime request/log archives, private machine paths, paper PDF/full text and private Git history are excluded. Public history retains the previously disclosed username-literal correction and the report24 local-path correction. The report24 publish shell continued after a failed privacy assertion; the current manifest is corrected, and history is retained. No model parameters or credentials were identified in that incident.

`/workspace/ViSkill-local-repro`, `/data/local` and `gpu-host` are deployment placeholders. Historical process IDs and receipts are evidence, not active endpoints. Some source references point to excluded runtime files. Future major results, including negative results, will be committed and pushed with the same public-data boundaries.

Upstream licenses are retained: [MIT LICENSE](LICENSE), [bundled veRL license](verl/LICENSE).
