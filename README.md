# ViSkill local reproduction and audit

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

This sanitized snapshot includes code, configs, synthetic map manifests/images, episode outcomes, audit scripts and compact numeric/update/sync/exit evidence. Model weights, checkpoints, complete runtime request/log archives, private machine paths, paper PDF/full text and private Git history are excluded. Earlier public history retains the previously disclosed username-literal correction; this update does not rewrite that history.

`/workspace/ViSkill-local-repro`, `/data/local` and `gpu-host` are deployment placeholders. Historical process IDs and receipts are evidence, not active endpoints. Some source references point to excluded runtime files. Future major results, including negative results, will be committed and pushed with the same public-data boundaries.

Upstream licenses are retained: [MIT LICENSE](LICENSE), [bundled veRL license](verl/LICENSE).
