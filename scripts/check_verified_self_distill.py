"""One verifier-assisted correction of the already frozen eligible trajectory."""
import json, os, time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

def main():
    root = Path(os.environ['TASK_ROOT'])
    source = root/'artifacts/component_tests/onpolicy_structured_v1'
    out = root/'artifacts/component_tests/verified_self_distill_v1'
    out.mkdir(exist_ok=False)
    record = json.loads((source/'collect_6/episode.json').read_text())
    previous = json.loads((source/'strategy_6.json').read_text())
    state = record['initial_env_state']
    player = list(state['player_pos'])
    goal = next([i, j] for i, row in enumerate(state['desc']) for j, cell in enumerate(row) if cell == 'G')
    actions = record['trajectory_actions']
    assert player == [0, 0] and goal == [1, 0] and len(actions) == 1 and record['traj_success']
    feedback = {'player_row_col': player, 'goal_row_col': goal,
                'goal_relative_to_player': 'below', 'goal_cardinally_adjacent': True,
                'executed_actions': actions, 'success': record['traj_success'],
                'source': 'saved training environment records, NOT model perception'}
    prompt = ('Correct your previous strategy using the attached initial image and the verified feedback below. '
              'The previous bottom-right goal claim is false. Generate one reusable conditional strategy '
              'inside <strategy>...</strategy>. WHEN must state the verified relative goal condition; '
              'WHAT must retain the observed action direction and count. Do not invent a longer route, '
              'diagonal action, or additional observations. Return the corrected strategy only.\n'
              'Previous model output:\n' + previous['strategy'] + '\nVerified feedback:\n' + json.dumps(feedback))
    set_seed(42)
    processor = AutoProcessor.from_pretrained(root/'models/b0', local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/b0', local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'}).eval()
    serialized = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'text', 'text': prompt}, {'type': 'image'}]}], tokenize=False, add_generation_prompt=True)
    (out/'prompt.txt').write_text(serialized)
    image = Image.open(source/'collect_6/frame_0.png').convert('RGB')
    batch = processor(text=[serialized], images=[image], return_tensors='pt').to('cuda')
    n = batch.input_ids.shape[1]
    start = time.time()
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=256, do_sample=False, repetition_penalty=1.05)
    response = processor.tokenizer.decode(output[0, n:], skip_special_tokens=True)
    result = {'raw_response': response, 'feedback': feedback, 'previous_strategy': previous,
              'model_calls': 1, 'paper_exact': False, 'training': False,
              'semantic_review_required_before_reuse': True, 'wall_seconds': time.time()-start,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
