# ViSkill local reproduction and audit

A local, no-external-model-API reproduction workspace for **ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills**. This is an independent reproduction and audit, not the official repository.

**Current status: PARTIAL.** Real full-parameter PPO actor/critic updates, checkpoint saving, and independent model reload have been verified. The single PPO batch admitted **zero new skills**, so policy/skill joint evolution has **not** passed. This is not a reproduction of the paper's performance numbers.

## Review these first

- [Current result and limitations](reports/02_ppo_smoke_test.md)
- [Paper/code audit](docs/01_paper_code_audit.md)
- [Distillation diagnosis and retained negative results](reports/04_distillation_diagnosis.md)
- [Explicit directional-variant component gate](reports/E_VARIANT_GATE.json)
- [Local evaluation](reports/01_local_evaluation.md)
- [Full chronological status](reports/REPRODUCTION_STATUS.md)
- [Reproduction manifest](reproduction_manifest.json)

Earlier chronological reports include historical states; the current PPO report and this page take precedence for the latest result.

## Verified results

| Check | Observed result | Scope |
|---|---|---|
| Base / released / released + released library | 5/20, 3/20, 20/20 successes | Small fixed-seed engineering evaluation only |
| Original free-text self-distillation | Semantic failures retained | Not passed |
| Explicit directional-contract variant | Two skills admitted; three actual retrieval uses succeeded in a component run | Modified algorithm, limited component qualification |
| Full-parameter PPO | One batch of 8; two optimizer mini-batches per rank; actor/critic parameter deltas finite and nonzero | Actual training, not import-only evidence |
| Skill update within that PPO batch | 0 admissions; 1/8 successful trajectories | Full joint-update acceptance failed |
| Independent model reload | Actor and critic loaded in a fresh process; validation 2/2 | Weights and empty skill sidecar only; no optimizer continuity claim |

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

This is a sanitized source snapshot. `/workspace/ViSkill-local-repro` and `/data/local` are deployment placeholders: adapt them to your machine before using the experiment-specific wrappers. Stored PID/timestamp receipts are historical evidence, not runnable launch instructions. Preserve the documented gates and budgets. The paper configuration is an archival mapping of shell overrides, not an executable training config.

Model weights, checkpoints, caches, full runtime logs, private machine snapshots, and the paper PDF/full text are excluded. Small synthetic FrozenLake artifacts and selected numeric evidence are included. Some historical evidence references point to excluded local artifacts; they must not be read as files included in this public snapshot. See [publication manifest](PUBLIC_RELEASE.json).

Upstream license and attribution files are retained, including [MIT LICENSE](LICENSE) and [bundled veRL license](verl/LICENSE).
