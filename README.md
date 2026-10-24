# ViSkill local reproduction and audit

A local, no-external-model-API reproduction workspace for **ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills**. This is an independent reproduction and audit, not the official repository.

**Current status: PARTIAL.** The completed 128-episode frozen capacity run produced 15 successes, 8 quality-eligible trajectories, 5 valid distilled strategies and 3 admitted skills. All 3 admitted skills were single-step; all 3 eligible multi-step trajectories failed distillation. A further 96 bounded diagnostic calls produced no intervention meeting its predeclared qualification gate. A prospective 64-episode paired check of corrected dynamics wording then yielded 2/32 versus 3/32 successes, with 2/32 quality-eligible in both arms; its predeclared improvement gate also failed. The prompt branch is closed. No additional PPO was run. This is not a reproduction of the paper's performance numbers.

## Review these first

- [Complete results summary and evidence limits](reports/06_results_summary.md)

- [Latest P0, 128-episode results and 96-call follow-up analysis](reports/05_mechanism_p0.md)
- [Earlier PPO and independent reload evidence](reports/02_ppo_smoke_test.md)
- [Paper/code audit](docs/01_paper_code_audit.md)
- [Distillation diagnosis and retained negative results](reports/04_distillation_diagnosis.md)
- [Explicit directional-variant component gate](reports/E_VARIANT_GATE.json)
- [Local evaluation](reports/01_local_evaluation.md)
- [Full chronological status](reports/REPRODUCTION_STATUS.md)
- [Reproduction manifest](reproduction_manifest.json)

Earlier chronological reports include historical states; the latest P0 report and this page take precedence for the latest result.

## Verified results

| Check | Observed result | Scope |
|---|---|---|
| Earlier full-parameter PPO | One 8-episode batch, actual actor/critic updates and independent weight reload | Training batch admitted 0 skills; empty library recovery only |
| P0 input/mask audit | Same skill image and processed pixels; correct generated/observation masks; distillation excluded from PPO response tokens | Real processor/worker preprocessing with explicitly scripted fixture outputs, not a new training claim |
| Frozen base-policy capacity | 128 episodes → 15 successes → 8 quality-eligible → 5 valid strategies → 3 admissions | 128 distinct initial boards; diagnostic library only |
| Multi-step capacity | 105 tasks → 9 successes → 3 quality-eligible → 0 valid strategies | All admitted skills remained single-step |
| Action-grounded prompt | 16 calls; image/text arms both 0/8 valid | Rejected, including regressions on previously correct single-step cases |
| Generic schema decoding | 16 calls; original prompt 5/8, grounded prompt 4/8 | Grounded arm: 2/3 multi-step but only 2/5 single-step; neither met 8/8 gate |
| Observation/example comparison | 64 single-turn calls on 16 previously failed states | Four paired arms, none met predeclared improvement gate; not 64 full episodes or held-out evaluation |
| Accurate dynamics description | 32 new boards × 2 arms = 64 full episodes / 240 calls; successes 2/32 → 3/32, quality-eligible 2/32 → 2/32 | 2 success improvements / 1 regression; failed predeclared gate, branch closed; correction remains opt-in |


The sole successful training trajectory contained three valid actions while its reconstructed shortest path required one. It failed the frozen quality ratio of 1. The acceptance threshold was not relaxed. A small two-episode validation result does not establish a learning benefit.

## Explicit deviations

Cold start is disabled and distillation uses the current local policy. The directional variant changes the strategy representation and uses privileged environment state for reject-only admission and retrieval checks. It also uses the documented state-progress reward and batch novelty fixes. These are explicit modifications, not claims of faithful original-algorithm reproduction. Original failures are preserved.

No full 300-step run, medical experiment, external inference API, or telemetry upload was performed for this reproduction.

## Source and environment

- [Official ViSkill repository](https://github.com/ZJU-REAL/ViSkill), base commit `7acaff880871b8d898d91c515c151080eabcba20`.
- [Paper](https://arxiv.org/abs/2610.12403).
- [Original upstream README](README_UPSTREAM.md).
- [Environment setup and version deviations](docs/03_environment_setup.md).
- [Frozen smoke protocol](configs/smoke/directional_ppo_v1.json) and [configuration](configs/smoke/frozenlake_tiny.yaml).

This update also restores bundled veRL model implementations and tests that the prior snapshot accidentally excluded with a broad `models/` ignore rule. Model-weight output exclusions are now root-anchored.

This is a sanitized source snapshot. `/workspace/ViSkill-local-repro` and `/data/local` are deployment placeholders: adapt them to your machine before using the experiment-specific wrappers. Stored PID/timestamp receipts are historical evidence, not runnable launch instructions. Preserve the documented gates and budgets. The paper configuration is an archival mapping of shell overrides, not an executable training config.

Model weights, checkpoints, caches, full runtime logs, private machine snapshots, and the paper PDF/full text are excluded. Small synthetic FrozenLake artifacts and selected numeric evidence are included. Some historical evidence references point to excluded local artifacts; they must not be read as files included in this public snapshot. See [publication manifest](PUBLIC_RELEASE.json).

Upstream license and attribution files are retained, including [MIT LICENSE](LICENSE) and [bundled veRL license](verl/LICENSE).
