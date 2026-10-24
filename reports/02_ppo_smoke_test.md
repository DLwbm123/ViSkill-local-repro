# Directional variant: one-batch PPO smoke

Status: **PARTIAL — real PPO update, model save and independent reload verified; joint skill update failed because admission was zero**.

Scope: explicit `directional_contract_v1` variant qualified by `E_VARIANT_GATE.json`. Original free-text strategy E remains PARTIAL. This is not a faithful reproduction claim or a performance experiment.

## Frozen run

- ID `directional_ppo_v1`; pro5000 physical GPUs **3,5**, isolated local Ray cluster.
- First launch Unix `1792820313.2427847`; all repairs share deadline `1792825713.2427847` (90 minutes). Original project start `1792772381.9046373`, total deadline `1792858781.9046373` unchanged.
- Base Qwen2.5-VL-3B, all parameters eligible (LoRA=0, vision tower unfrozen), actor/critic FSDP CPU parameter/optimizer offload. FP32 master weights, BF16 mixed precision and rollout. No reference worker because both KL terms are disabled, matching the official script.
- Exactly one training batch of 8, fixed seeds 1–8, mini-batch 4, per-GPU micro-batch 1. One batch contains two optimizer mini-batches; do not describe it as only one optimizer call. Validation seeds 9,10, no selection or tuning against their results.
- Empty library; no saved diagnostic skills used as cold start. Self-distillation by current local policy; no external inference/telemetry.
- Save actor/critic **model shards only**, dataloader and immutable library sidecar. No optimizer/RNG continuity claim. A second fresh worker/cluster loads this pair and executes validation only, with no second training batch.
- Baseline allocated storage 35,784,097,792 bytes; own conservative write stop 75,784,097,792 bytes (<project 85GB stop). Estimated model shards 28–32GB; no duplicate HF export or optimizer checkpoint.
- Supervisor tracks its unique environment token and process creation times, stops only its own processes on phase completion/error/hard limit, never invokes global `ray stop`. Process command lines audited for neutral names.

## Changes from official script

| Item | Official script | Smoke | Implication |
|---|---|---|---|
| GPUs / batch / mini-batch | 4 / 128 / 32 | 2 / 8 / 4 | Reduced engineering test |
| Training steps | 301 | 1 | No learning-curve claim |
| Train / val environments | Large sampled lists | Fixed 1–8 / 9–10 | No paper success-rate claim |
| Prompt / response caps | 5000 / 4000 | 8192 / 4096 | Explicit history capacity; 512 tokens per turn and 5 turns retained |
| Rollout memory fraction | .6 | .22 | Coexistence with unrelated workloads |
| Attention | native backend defaults | SGLang triton / vision sdpa | Compatible Blackwell execution path |
| Strategy | free-text | directional JSON with reject-only privileged-state admission/retrieval guard | Algorithm change; original failure retained |
| Reward | action-prefix skill_guided | state_progress_bounded | Previously documented reward variant |
| Novelty | code batch admission | recheck_batch_novelty | Previously documented batch duplicate correction |
| Save contents | model, hf_model, optimizer, extra | model | Weight/library recovery only |
| Library pairing | absent upstream | opt-in pair marker and snapshot | Explicit recovery repair |
| Logging | console, wandb | console/files only | No upload |

`configs/paper/frozenlake.yaml` archives all explicit original shell overrides (unresolved shell paths retained; not an executable configuration). The official script remains unchanged. `configs/smoke/frozenlake_tiny.yaml` is merged over bundled trainer defaults plus the named variant.

## Checks and failures before launch

- First compose used `file:` instead of `file://`; corrected Hydra search-path syntax.
- Variant YAML was missing on remote; synchronized the named config.
- Delta helper under `scripts` collided with an installed package; moved to `verl.utils.local_update_audit`.
- Bundled `validate_config` passed; actual dataset expansion verified train 1–8 and val 9,10. Upstream explicit seed-list requires a trailing sentinel, which is documented and excluded by its own slicing.
- CPU delta-audit self-check passed. It samples up to 4096 values per local trainable optimizer parameter before/after each real step, records finite gradient and delta per rank. This check is not PPO evidence. Model instrumentation is disabled unless `LOCAL_UPDATE_AUDIT_DIR` exists.

## Acceptance and evidence

PPO PASS requires real rollout/reward, finite actor/critic loss and backward, nonzero sampled parameter delta, new policy-derived skill admission, model save/load, matching pair restore, and validation. Zero admitted skills leaves the full-loop result incomplete; it does not authorize favorable seed replacement.

Remote evidence: `state/components/directional_ppo_v1_{receipt,EXIT}.json`, `logs/ppo_smoke/directional_ppo_v1_{train,reload}.log`, `results/ppo_smoke/directional_ppo_v1/`, `checkpoints/smoke/directional_ppo_v1/`, `artifacts/ppo_smoke/directional_ppo_v1/` under `/data/local/ViSkill/`. Exit and completion files must be read before reporting any success.

Latest startup verification: actor/critic models loaded, two native SGLang replicas bound to physical GPUs3 and5, `Training Progress 0/1` reached. `ps` confirmed neutral supervisor/Ray worker command lines. This establishes startup only.

## First batch results (2026-10-24 13:42 Shanghai)

Actual `run_ppo` returned for training in 234.07 seconds. Both actor and critic performed two optimizer mini-batches on each rank. All eight delta-audit records are finite and nonzero: actor local sample L2 range **0.000367848–0.000510382**, critic **0.003461609–0.005033503**. Each record sampled 282,624 values; this is sampled parameter evidence, not a full parameter checksum. Trainable local element counts summed across two ranks are 3,754,622,976 actor and 3,754,625,026 critic (including its value head).

Logged actor pg_loss=0.0338779, critic vf_loss=1.7594881; averaged actor grad norm=8.6136248, critic=218.6172924. Training success **1/8**, admitted skills **0**, retrieved skills **0**. This proves PPO updates but fails the planned joint skill-library update criterion. Fixed-seed post-update validation succeeded 2/2, which is only an engineering observation.

The one successful recorded rollout contains valid actions `Left, Left, Down`, interleaved with two invalid `None` answers. Among the eight fixed initial boards, only seed6 reaches the goal with that action sequence: two boundary no-ops then Down, versus shortest path 1. Thus its 3/1 action ratio exceeds the frozen quality_ratio=1 and correctly blocks self-distillation/admission. Seed attribution is a reconstruction from saved actions and prior fixed-board states: the upstream rollout logger omitted seed/state/distillation audit fields. Do not loosen the gate after observing this result. The generated reasoning also contains visibly garbled multilingual text; this is an observed anomaly, not proof of weight-sync corruption. Initial hybrid weight synchronization remains a targeted diagnostic question.

Both actor and critic actual model shards saved; pair marker records an **empty** library. Independent fresh-process reload is running, and must verify both models plus validation; restoring an empty pair cannot prove nonempty skill recovery. No further PPO batch is authorized by this run.

Training worker reported peak allocated 34.985GiB and reserved 40.109GiB (aggregated trainer metric, not per-GPU nvidia-smi peak), host used RAM metric63.306GiB. Four-root actual disk snapshot **68,346,335,232 bytes**, below75,784,097,792 stop. Full train log and model shards remain remote; small delta/rollout/pair evidence is copied under docs/evidence/ppo_smoke/directional_ppo_v1.

## Completed independent reload

Both phases exited0; supervisor reason=null, end Unix1792820802.6111708 (2026-10-24 13:46:42 Asia/Shanghai), total489.37s. Reload took190.835s in a new process/local Ray cluster. Actual actor and critic shards loaded at global_step1; validation-only then completed2/2 success with mean reward1.03, matching the earlier validation aggregate. Active restored skill index is readable and empty. This confirms model and empty-sidecar loading, not nonempty skill recovery or optimizer continuity. Final allocated-disk peak68,347,863,040bytes. No remaining process with this run token was found in the final check.

Frozen run remains PARTIAL: zero library update. Do not repeat it or append PPO updates. The next diagnostic should first preserve/audit saved trajectories and investigate the initial hybrid rollout text anomaly without treating it as established weight corruption. No new run was launched while preparing this summary.
