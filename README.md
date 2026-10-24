# ViSkill local reproduction and audit

Independent local reproduction of **ViSkill: Reinforcing VLM Agents with Evolving Visual-Native Skills**, using local inference with no external model API. This is not the official repository.

**Current result: online cards entered real PPO training; a learning benefit has NOT been established.** Both paired arms completed four32-episode training batches, pre/post64-map Dev evaluation and a final32-map probe:576 complete episodes and2056 model calls. One training seed per arm; no final weights retained.

| Dev metric | Both arms before training | A: ordinary PPO after | B: online cards, reward weight0, after |
|---|---:|---:|---:|
| Success |9/64|6/64|10/64|
| Shortest-path quality eligibility |8/64|5/64|10/64|
| Multi-step success |6/53|3/53|2/53|
| Multi-step quality eligibility |6/53|2/53|2/53|

B's own success improved by only1/64, with8 paired improvements and7 regressions. Its multi-step success fell from6/53 to2/53. The numerical B-A difference includes A's decline; supplementary paired-map intervals cross zero and do not cover training-seed variability. These results do not establish stable improvement, multi-step learning or the paper's performance.

B admitted5 skills:4 one-step and1 three-step. Ten later training samples actually used one-step cards and participated in actor/critic updates. The three-step card arrived in the last batch, was never subsequently injected, and had no applicable candidate among78 multi-step Dev-after/probe tasks. Actual skill-reward weight was0; all shadow bonuses were0. Complete F remains PARTIAL because reward training and independent nonempty-library/model restoration were not verified.

## Review these first

- [Complete paired training results and evidence limits](reports/11_bootstrap_pair_results.md)
- [All experiment results summary](reports/06_results_summary.md)
- [576 episode outcomes](docs/evidence/components/bootstrap_card_pair_v1/all_episode_outcomes.json)
- [29 raw distillation outputs](docs/evidence/components/bootstrap_card_pair_v1/all_distillation_outputs.json)
- [Final five-card library](docs/evidence/components/bootstrap_card_pair_v1/B/library/skill_index.json)
- [Paired uncertainty estimates](docs/evidence/components/bootstrap_card_pair_v1/paired_uncertainty.json)
- [Frozen protocol and exact seeds](configs/smoke/bootstrap_card_pair_v1.json)
- [Lifecycle fault and resource-gate recovery](reports/09_bootstrap_incidents.md)
- [Full chronological status](reports/REPRODUCTION_STATUS.md)

Chronological reports and old gate snapshots retain historical states. This page and report11 take precedence for current results. No completed batch or inference request was repeated to recover the interrupted lifecycle cleanup or the unstarted B resource gate.

## Earlier evidence retained

The earlier128-episode capacity run produced15 successes,8 quality-eligible trajectories,5 valid strategies and3 admitted one-step diagnostic skills. All3 eligible multi-step trajectories failed distillation. Another96 bounded prompt/schema/observation calls met no qualification gate. A separate64-episode dynamics-wording check had2/32 versus3/32 successes and2/32 quality-eligible in both arms; its improvement gate failed. Those protocols remain closed, and the old3-skill diagnostic library is separate from the new5-skill online library.

Earlier one-batch PPO and independent model reload established real updates and empty-library restoration. P0 checked real processor/worker image and mask handling using explicitly scripted fixtures. Neither substitutes for a scientific learning-benefit claim. See [P0/capacity report](reports/05_mechanism_p0.md) and [earlier PPO report](reports/02_ppo_smoke_test.md).

## Explicit algorithm and runtime boundaries

The named directional variant changes the strategy representation and uses privileged deterministic state for reject-only semantic/admission/retrieval checks; reward and batch-novelty deviations are documented. Cold start is disabled and current-policy self-distillation is retained. Original free-text failures remain recorded. The paired pilot enabled accurate dynamics wording equally in both arms and used no schema/prompt search or curriculum.

A confirmed concurrent RGB-environment shutdown stall was repaired without repeating its54 completed requests, rewriting its failed episode or resetting clocks. B's resource gate was preserved; only its unstarted arm resumed after allowed GPU memory recovered. Model calls, failures and costs remain in the evidence. Monitored peak additional storage was about80.37MB, below the1.8GB write stop and2GB budget. Old checkpoints were retained; the new pilot deliberately saved no final model, optimizer or RNG state.

## Source and public scope

- [Official repository](https://github.com/ZJU-REAL/ViSkill), pinned upstream7acaff880871b8d898d91c515c151080eabcba20.
- [Paper](https://arxiv.org/abs/2610.12403), [paper/code audit](docs/01_paper_code_audit.md), [upstream README](README_UPSTREAM.md).
- [Environment and version deviations](docs/03_environment_setup.md), [publication manifest](PUBLIC_RELEASE.json).

This sanitized source snapshot includes code, configs, exact synthetic map manifests, compact numeric outcomes, raw distillation outputs, five small cards and selected optimizer/sync/exit evidence. Model weights, checkpoints, full runtime request/log archives, machine-private paths, paper PDF/full text and private local Git history are excluded. Earlier public history retains a previously disclosed username-literal correction; this update does not rewrite that history.

`/workspace/ViSkill-local-repro`, `/data/local` and `gpu-host` are deployment placeholders. Adapt them and provide the documented local weights/environment before using machine-specific wrappers. Historical PIDs, ports and receipts are evidence, not active endpoints. Some source references lead to excluded raw runtime artifacts.

Upstream licenses are retained: [MIT LICENSE](LICENSE), [bundled veRL license](verl/LICENSE).
