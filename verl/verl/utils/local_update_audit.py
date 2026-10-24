"""Opt-in sample delta around real optimizer steps; not a model checksum."""
import json
import os
from pathlib import Path
import torch


def snapshot(optimizer):
    if not os.environ.get('LOCAL_UPDATE_AUDIT_DIR'):
        return None
    entries = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad and p.numel():
                flat = p.detach().view(-1)
                stride = max(1, flat.numel() // 4096)
                entries.append((p, stride, flat[::stride][:4096].float().cpu().clone()))
    return entries


def record(role, entries, grad_norm):
    if entries is None:
        return
    total = 0.0
    finite = bool(torch.isfinite(grad_norm).all())
    for p, stride, before in entries:
        after = p.detach().view(-1)[::stride][:4096].float().cpu()
        finite = finite and bool(torch.isfinite(after).all())
        total += float((after - before).double().square().sum())
    folder = Path(os.environ['LOCAL_UPDATE_AUDIT_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    row = {'role': role, 'pid': os.getpid(), 'rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
           'sample_delta_l2': total ** .5, 'grad_norm': float(grad_norm), 'finite': finite,
           'sampled_values': sum(x.numel() for _, _, x in entries),
           'trainable_local_elements': sum(p.numel() for p, _, _ in entries)}
    with (folder / f'{role}_{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    assert finite, 'non-finite optimizer audit'
