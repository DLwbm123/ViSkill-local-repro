"""Full saved collection census after dispatch failure; no model or input replay."""
import os,json,runpy,time,signal
from pathlib import Path
ROOT=Path(os.environ['TASK_ROOT']);BASE=ROOT/'artifacts/component_tests/support_matched_curriculum_v1';DEADLINE=1792885281.834
assert time.time()<DEADLINE
signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('fixed failure census deadline')));signal.alarm(max(1,int(DEADLINE-time.time())))
helper=runpy.run_path(str(ROOT/'src/scripts/summarize_value_head_single_step.py'),run_name='paired_helpers')
rows=helper['episodes'](BASE/'collect');manifest=json.loads((BASE/'data_manifest.json').read_text());expected={r['seed']:r for r in manifest['splits']['train']};assert len(rows)==40 and {r['seed'] for r in rows}==expected.keys()
all_raw={json.loads(f.read_text())['seed']:json.loads(f.read_text()) for f in (BASE/'collect/episodes').glob('*/episode.json')}
requests=[json.loads(f.read_text()) for f in (BASE/'collect/requests').glob('*.json')];assert len(requests)==int((BASE/'collect/model_calls.txt').read_text())==134
for row in rows:
    source=all_raw[row['seed']];assert source['initial_env_state']==expected[row['seed']]['state'] and row['category']==expected[row['seed']]['category']
    row.pop('episode_path');row['distillation_audit']=source['distillation_audit']
source=json.loads((ROOT/'results/ppo_smoke/bootstrap_card_pair_v1/B/library/skill_index.json').read_text());current=json.loads((BASE/'collect/context_libraries/cards/skill_index.json').read_text());assert source==current
counts={}
for category in ['all','conditional','medium','single','long']:
    use=[r for r in rows if category=='all' or r['category']==category]
    counts[category]={'episodes':len(use),'success':sum(r['success'] for r in use),'quality':sum(r['quality'] for r in use),'Stop_turns':sum(r['stop_turns'] for r in use),'Stop_episodes':sum(r['has_stop'] for r in use),'holes':sum(r['hole'] for r in use)}
result={'status':'COMPLETE_POSTHOC_FAILURE_CENSUS','preparation_started_unix':1792884681.834,'deadline_unix':DEADLINE,'completed_unix':time.time(),'maximum_new_audit_bytes':2000000,'model_forwards':0,'model_backwards':0,'model_optimizer_steps':0,'environment_steps':0,'new_model_dispatches':0,'actual_saved_collection':{'episodes':40,'rollout_calls':sum(r['kind']=='rollout' for r in requests),'distill_calls':sum(r['kind']=='distill' for r in requests),'total_calls':len(requests)},'counts':counts,'records':rows,'library_json_unchanged':True,'admissions':0,'exact_processed_bank_preserved':False,'new_training_steps':0,'new_Dev_episodes':0,'frozen_scientific_gate_evaluated':False,'cause':'Inherited train-phase distill.mode=self permits independent self-distillation; the no-distill assertion runs only after all40 trajectories. Two attempts occurred. Assertion stopped before private bank persistence and before any actor scoring/update.','decision':'Close the dispatched pair as engineering/protocol failure; preserve all40 outcomes/134 calls, no repeated sampling or reconstruction-as-original-input training.','scope':'Posthoc full census of completed collection, not a heldout performance comparison or a scientific rejection of support matching.'}
text=json.dumps(result,indent=2);assert len(text.encode())<2000000;(BASE/'failure_audit.json').write_text(text);signal.alarm(0);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
