# Local self-distillation diagnosis

## Scope and prior failure

C/D engineering validation passed; the 20-episode base/released/released+library successes were 5/3/20. These are not paper-scale results. E remains partial because a valid strategy tag does not establish a correct WHEN/WHAT condition or action plan. The released checkpoint also copied a false adjacent-goal caption while completing the card-guided fixture. Therefore card reuse and caption correctness must be evaluated separately. F has never run.

Repeated unchanged monitoring did not narrow this cause. The user's new request authorizes finite causal diagnostics, superseding the agent's earlier stop on additional prompt variants. Original failures, GPU3/4/5 boundary, storage ceiling, original 24-hour clock, and D/E prerequisite remain intact. The old six-hour preparation attempt remains closed.

## Nine-condition causal diagnostic

Protocol `configs/smoke/distill_causal_v1.json`; one pinned base model, Transformers BF16 SDPA, original fixture, nine predeclared greedy generations, no training or new seeds. EXIT0. Original setup start1792772381.9046373 retained; local15-minute timeout. Full results: `artifacts/component_tests/distill_causal_v1/results.json`.

| Question | Observation | Supported conclusion |
|---|---|---|
| Are images missing? | Original120 has279 visual tokens, matching grid; visual encoder called once, output279×2048. Blank same-sized images give relative embedding L2=3.0498, cosine=.1371 and a different response. | Images reach the encoder and affect this inference; not evidence of an absent image path. This does not prove correct visual understanding. |
| Is thumbnail resolution sufficient explanation? | 120→256 increases visual tokens279→818; false adjacent-goal example persists at256. | Increasing resolution alone is insufficient; do not adopt it as a fix. |
| Does the forced strategy prefix cause the error? | Removing prefix leaves the120 response unchanged and still copies the example at256. | Prefix removal alone is insufficient. Prefix is also used by official code. |
| Is example copying the only problem? | Removing examples at256 produces an illegal diagonal up-left strategy. | Example copying is observable, but removing examples does not restore grounded action semantics. |
| Can the base identify the board? | Initial-image probe returns player[0,1], goal[2,2], holes[3,3], whereas actual player[0,0], goal[3,3], holes[1,1]/[1,2]. | State perception is wrong in this direct probe. |
| Can it read the trajectory? | Large-card transcription gives down,right,down,right,right,right; actual second action is down. Start/end corners are described correctly. | Temporal/action reading is imperfect despite some correct global visual facts. |
| Is visual perception the only issue? | Symbolic correct state/actions still produces wrong coordinate updates and only five actions. | Coordinate-heavy language synthesis also fails; cannot blame renderer alone. |

The initial image was visually inspected and agrees with fixture metadata. No file corruption, blank card or swapped image order was found. These finite diagnostics do not establish the model's general incapability, nor authorize replacing the model, deleting distillation, or lowering E acceptance.

## Next discriminating test

`distill_decomposition_v1` predeclares five calls: exact action echo, consecutive-run compression, board perception with sprite legend, seven separate full-sized frames, and two-stage strategy synthesis using the local policy's raw action compression plus original frames. No manually corrected model output or hand-written strategy is inserted. This separates elementary instruction-following from coordinate/trajectory synthesis and tests an explicit non-paper-exact inference variant. Semantic review and real card reuse are required before considering any E acceptance; no condition independently releases PPO.

## Decomposition result and bounded structured-cycle test

Five-call decomposition EXIT0. Exact action echo and run-length compression are correct (down3/right3). Sprite legend improves goal localization but player and one hole remain wrong. Seven full frames still yield an incorrect inferred action sequence, although top-left/bottom-right endpoints are correct. Two-stage synthesis using raw current-policy action compression gives correct directions/counts but omits WHEN. Thus elementary copying/compression works; spatial/temporal reasoning and instruction coverage fail under the original task.

Next protocol `structured_distill_cycle_v1` uses three current-policy generations: visual condition in corner/edge language, action compression from recorded actions, explicit conditional synthesis. Original image and raw intermediates are retained; no manual correction. It then renders, inserts into an isolated empty fixture library, reloads, retrieves and runs at most5 real reuse turns. The user-requested causal diagnostic deadline1792815515.061637 is shared, not restarted. This is an explicit non-paper-exact inference workflow; success would establish a candidate repair on one fixture, not base learning or a general semantic verifier.

## Structured cycle, applicability falsification, and action adherence

Three-stage current-base-policy generation produced correct top-left/bottom-right visual facts, down3/right3 compression, and explicit WHEN/WHAT text. Card rendering, isolated-library admission, disk reload, retrieval(score.95), and actual base reuse succeeded in4 turns with11 executed actions (5 boundary no-ops). The reasoning still mentioned illegal diagonal movement. This is a candidate repair on one fixture, not general grounding success. The requested hole-free-route clause was omitted.

A frozen two-map CPU diagnostic moved one hole from(1,1) to(1,0), preserving player/goal and hole count. Retrieval remains accepted at.80. Executing the reference actions described by the generated strategy succeeds on the original but falls into the hole on the first Down in the counterfactual. This is reference-route falsification, not policy sampling. It shows the corner-only WHEN is insufficient and approximate retrieval is not a route-validity check.

Two additional fixed model calls compared an explicit PROVIDED THAT clause and structured applicability fields. The model can now express a clear-route requirement; the JSON field remains generic about which path. Actual policy adherence was tested using the predeclared prose output, unchanged, on both frozen maps. Both failed: original Down,Right enters(1,1); counterfactual Down enters(1,0). This change also altered the action formatting in the card to JSON-like text, so it is not a clean causal estimate of the clause alone. It does establish that satisfying a textual condition-field check is insufficient. No threshold was tuned to hide this negative result.

No additional fixture prompt search is planned. The next finite qualification switches to actual current-policy trajectories: fixed training seeds1–8, all collected before any admission, original-vs-three-stage strategies generated on the same quality-successful trajectories, structured candidate library mechanically admitted/reloaded, all8 states reused if nonempty, utility updated. No held-out evaluation seeds, solver trajectories, warm library, or training. Empty/zero-eligible batch stops without searching for favorable seeds. Semantic review remains mandatory before considering integration into the official agent loop.

## Genuine on-policy qualification and serialization repair

Fixed seeds1–8 collection produced2 successes (6 and7), but quality ratio1 admitted only seed6. Its actual goal is(1,0), one step below player(0,0); the generated visual condition incorrectly says bottom-right. Original strategy proposes diagonal movement; structured strategy preserves Down1 but still has wrong WHEN. Seed7 is successful with boundary no-ops and fails the quality gate, so it was not selected for distillation. This is a genuine cold_start=false current-policy data flow, with no solver trajectory.

First save failed because the earlier explicit positions extension retained NumPy int64 coordinates that JSON cannot serialize. This is a bug in our positions-support extension, not evidence that the untouched upstream always fails. Repair converts only integer coordinates via operator.index (floats are rejected), serializes before publication and atomically replaces the index. A regression test verifies NumPy roundtrip and that a failed save leaves the previous index readable;1 PASS. It does not claim atomicity of the entire image/index directory. Original truncated index and EXIT1 remain.

Recovery loaded saved collection frames/metadata and saved strategy without regenerating either. Its first entry failed because JSON restored player_pos as a list, whereas BFS expects a hashable tuple; the restore boundary now normalizes that type. This separate recovery-script failure is retained. The next recovery completed before the original online-phase deadline:1 skill admitted, library reloaded,8 reuse episodes,2 successes,3 skill uses; final utility.5355, use_count3. Reuse had not begun before failure; because the prior RNG state was not saved, reuse was explicitly seeded42, not claimed a bit-identical continuation. No training occurred.

This proves a real mechanical policy→success→self-generated strategy→card→admission→reload→retrieval→utility cycle. It does not prove strategy semantics or official training-loop integration. No measured reuse benefit is claimed from2/8 vs2/8, especially with the separately declared reuse RNG.

## Random-layout grounding and processor parity

Eight saved training images were queried under the existing corner/edge prompt and a relative-goal prompt (all retained, no selection). The base shows stereotyped corners even for random placements. Relative direction is0/8 correct; adjacency3/8. The eligible seed6 synthesis then claims adjacency to a blue object, so it remains invalid. This is a probe-specific result, not a claim about all model spatial capabilities.

A fixed slow-processor control reuses the exact serialized requests, images and weights. All token IDs and visual grids match the fast processor; relative pixel L2 is approximately0.00018–0.00031. Slow processing is also0/8 on direction and repeats the original incorrect fixture strategy. The automatic fast-processor migration does not explain these failures in this controlled test; no processor change is adopted.

## Verifier-assisted correction and fixed input factorial (completed)

`configs/smoke/verified_self_distill_v1.json` freezes a single verifier-assisted correction on the existing quality-successful trajectory. The verifier uses recorded environment facts to expose incorrect WHEN/WHAT, and the same local policy gets exactly one correction request. A successful correction would be tested through the fixed reuse set; a failure stops this variant. This is explicitly non-paper-exact because verified state feedback supplements visual input. It must not erase original failures or release PPO without the remaining E integration checks. No additional blind prompt/seed/model search is planned.


The predeclared single correction completed EXIT0, one model call. Explicit saved-state feedback supplied player(0,0), goal(1,0), adjacent/below and the actual Down1 success. The policy repeated the old wrong WHEN verbatim. Semantic review FAIL; no corrected skill was admitted, and conditional reuse was not run. This variant is closed without a second correction.

A separate fixed 2×2 attribution audit then removed only the prior answer body and/or image (three new calls, original image+old cell reused). The exact original serialized request was reconstructed and asserted before inference. All task wording and verifier feedback were held fixed; image-absent cells still retain the request's image reference, so this is an input-removal contrast rather than an optimized text-only task.

| Image | Old answer | Observed output |
|---|---|---|
| Present | Present | Exact wrong bottom-right route copied |
| Absent | Present | Same exact wrong route copied |
| Present | Withheld | Correct relative below relation, but invents player bottom-right/(2,3) |
| Absent | Withheld | Correct below relation, but calls row0 the bottom row and omits explicit count |

For this fixed case, old-answer copying persists without an image. Withholding the old answer changes the output but does not yield a fully correct strategy. Thus old-answer anchoring, spatial/coordinate semantics and condition fidelity all matter; image processing alone is insufficient. This tiny deterministic contrast does not quantify population effects. No factorial output was admitted or used for PPO.

## Practical next route and limits

Do not spend the remaining budget on another free-text paraphrase of this fixture. The next useful engineering decision is to separate structured factual/action fields, semantic validation and executable applicability from free-form caption generation, and inspect how such a candidate would enter the official callback. Any prototype must be explicitly recorded as a changed method, preserve the original one-stage failure, and cannot use a handwritten final strategy or solver trajectory to satisfy the original E gate. Before GPU execution, freeze a minimal acceptance test on the existing training records; do not expand seeds/models, use evaluation answers, or auto-release PPO. If correctness would require verified environment facts, retain that privileged-input limitation rather than claiming visual-native self-distillation succeeds.

All jobs in this diagnosis have exited. Final remote budget audit:35,779,014,656 conservative allocated bytes, including8KiB legacy cache allowance; no project process remains. Only physicalGPU4 was used, torch peak in the real loop approximately7.73GB. Original total deadline1792858781.9046373 is unchanged. Reports and failures are retained locally; no remote publication.


## Structured contract qualification (2026-10-24 heartbeat)

The predeclared structured_contract_v1 tested one new representation on the same previously eligible seed6. Input: original initial/terminal images, observed successful actions, and success flag; no board coordinates, environment desc, old answer or verifier answer supplied to the model. Output schema separates an integer goal offset, a clear-route requirement and action/count entries. A diagnostic-only validator checks observed displacement/transitions, action equality and deterministic reference-route applicability; six synthetic validator tests pass. An initial test-import namespace collision was fixed and its failed collection log retained. The helper does not alter default upstream behavior.

The actual one-call model result EXIT0 was fenced JSON with WHAT Down1 but WHEN goal_offset[-1,0], whereas the observed successful motion/goal offset is[1,0]. The strict parser rejects the fence; an offline-only view of its JSON body independently confirms the sign error. Thus merely accepting markdown would not fix the candidate. The closed protocol stops with0 admission and0 reuse. No official callback or agent-loop execution of this candidate is claimed; their code path remains conditional and was not reached. The validator uses privileged state and any future guarded retrieval must disclose this limitation.

A separate exact-request arithmetic contrast contract_numerics_v1 tested the same saved prompt/images at FP32/eager against the retained BF16/SDPA output. One call only, no new prompt or admission. Precision and attention implementation change together, so any difference would not isolate either factor. This checks a possible numerical implementation explanation on the new GPU architecture before attributing all failure to policy semantics.


The FP32/eager contrast completed EXIT0 and returned exactly the same fenced JSON, including the incorrect[-1,0] displacement. Peak torch allocated15,128,573,440 bytes; one inference4.02s. This rules out changing both numerical precision and fused attention as a sufficient repair for this fixed failure; it does not prove that every numerical path is correct. A nonfatal kernel-cache-directory warning was retained in the remote log. No candidate was admitted or trained.

Next single predeclared diagnostic: configs/smoke/structured_checkpoint_contrast_v1.json (NOT RUN). Compare the existing released weights against the saved base output on this identical structured request, after checking processor compatibility. This asks whether the already trained policy supplies the missing structured grounding. It changes neither seeds nor prompts, downloads no model, and does not substitute the released-policy result for base E qualification. Do not resume already closed correction/structured/numerics attempts.


## Directional representation and actual official-loop qualification

The pinned released checkpoint exactly reproduced the base signed-offset error. Effective tokenizer vocab/special tokens, generation configs, model dimensions and processed token/image tensors were checked equal before that one-call contrast. No released result is substituted for base qualification.

A predeclared representation ablation replaced signed goal offsets with named direction and Manhattan distance, keeping the same initial/terminal images and actual successful actions. Exactly one current-base-policy call generated down/1 and Down1 correctly. A predeclared lossless single outer JSON-fence normalization was allowed; no JSON value was edited. Actual callback admission, save/reload and8 wrapper reuse episodes succeeded mechanically:3 task successes,2 applicable skill uses both successful, utility.595. Query seed5 scored.8375 but its condition mismatched and the privileged-state guard refused injection. No performance advantage is inferred.

The optional directional_contract_v1 representation and retrieval.contract_guard now have real integration hooks in GymAgentLoop, DistillCallback and ViSkillProvider, with free_text/unguarded defaults unchanged. The environment skill-state metadata now includes explicit is_slippery/slip_prob; unknown/stochastic dynamics fail the optional guard. Model generation receives images, recorded actions and success only, not environment-state answers. Verifier/retrieval guards use privileged state, so this is an explicit changed method. Runtime docs/config live in configs/low_resource/directional_skill_system.yaml. Nineteen related checks passed; a later single-role-envelope test adds one distinct check, with the9 affected parser/integration checks passing after that change.

The full official GymAgentLoop.run was exercised on all8 fixed training seeds from an empty library using a local Transformers token adapter (not mocked policy replies):1 collection success/1 admission, then8 reuse/2 successes/1 skill use, utility.55. Two distillation calls yielded1 valid strategy; invalid output was rejected. The first entry failed before loading/rollout due to missing cachetools, fixed with pin5.5.2; the restarted worker shared the first15-minute deadline. All outputs and the original failure remain.

Native SGLang qualification invoked the actual bundled SGLangHttpServer.generate method on a local Engine handle and the full official GymAgentLoop. It does not prove Ray scheduling or FSDP/rollout weight synchronization. Eight collection episodes produced3 quality successes. All3 raw responses began with an extra assistant line; the strict parser rejected them, so no reuse started. Offline review found left1 and down1 semantically valid, but up/up/right had wrong direction/distance. A separate default-off allow_assistant_envelope flag removes at most one exact outer role line before fence parsing; it changes no field and rejects additional prose/messages. The origin of that framing is not established as a backend bug versus model emission.

Recovery reused all saved collection records and raw model outputs without recollection or redistillation, in a new isolated test library. Original failed raw outputs remain. Exactly2 skills admitted; the wrong multi-step condition stayed rejected. The previously unstarted native reuse phase was explicitly initialized with engine seed42, not claimed RNG-continuous recovery. Eight reuse episodes gave4 task successes;3 actual skill injections (seeds2/6/7) all succeeded with the generated Left1/Down1 actions. Persisted utilities are.55/count1 and.595/count2; reward scores for these are1.02. Another new invalid multi-step distillation was rejected. EXIT0 and native processes exited. Card images were opened and visually checked. No policy/critic training has occurred.

This closes the scoped E component gate for the explicitly modified directional variant, not the original free-text algorithm and not a general spatial-reasoning or performance claim. Original E remains partial/negative. Only a clearly labelled one-step full-parameter PPO smoke of this variant may be prepared next, still requiring actual actor/critic backward, nonzero parameter delta, real checkpoint reload and skill/policy pairing. Native Ray/FSDP synchronization and all F requirements remain unverified.
