"""Diagnostic-only initial request capture; no training or distillation."""
import json
import os
from pathlib import Path
from vagen.agent_loop.gym_agent_loop_skill import GymAgentLoop


class RequestProbeLoop(GymAgentLoop):
    async def run(self, sampling_params, **kwargs):
        encoded_seed = int(kwargs['seed'])
        sampled = encoded_seed >= 100
        seed = encoded_seed - 100 if sampled else encoded_seed
        assert 1 <= seed <= 8 and kwargs['max_turns'] == 1
        kwargs['seed'] = seed
        params = dict(sampling_params, temperature=1.0 if sampled else 0.0)
        root = Path(os.environ['TASK_ROOT']) / 'artifacts/component_tests/hybrid_request_probe_v1'
        folder = root / 'hybrid' / f'{"sampled" if sampled else "greedy"}_{seed}'
        folder.mkdir(parents=True, exist_ok=False)
        manager = self.server_manager
        tokenizer = self.tokenizer

        class Recorder:
            calls = 0

            async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None):
                self.calls += 1
                assert self.calls == 1 and image_data and len(image_data) == 1
                server = manager._choose_server(request_id)
                image_data[0].save(folder / 'input.png')
                row = {'seed': seed, 'temperature': params['temperature'], 'prompt_ids': list(prompt_ids),
                       'sampling_params': dict(sampling_params), 'image_count': len(image_data),
                       'server_handle': str(server), 'prompt_text': tokenizer.decode(prompt_ids, skip_special_tokens=False)}
                (folder / 'request.json').write_text(json.dumps(row, indent=2))
                output = await manager.generate(request_id, prompt_ids=prompt_ids,
                                                sampling_params=sampling_params, image_data=image_data)
                row['output_ids'] = list(output.token_ids)
                row['output_text'] = tokenizer.decode(output.token_ids, skip_special_tokens=True)
                (folder / 'result.json').write_text(json.dumps(row, indent=2, ensure_ascii=False))
                return output

        self.server_manager = Recorder()
        result = await super().run(params, **kwargs)
        assert self.server_manager.calls == 1
        return result
