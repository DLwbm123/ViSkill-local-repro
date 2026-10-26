"""Prepare the four local tools; search remains explicitly deferred."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import venv
import zipfile

ROOT = Path(os.environ['TASK_ROOT'])
P = json.loads((ROOT / 'plan.json').read_text())
LOCK = threading.Lock()
PROGRESS = {}


def progress(name, **values):
    with LOCK:
        PROGRESS[name] = dict(updated_unix=time.time(), **values)
        tmp = ROOT / 'state/progress.tmp'
        tmp.write_text(json.dumps(PROGRESS, indent=2))
        tmp.replace(ROOT / 'state/progress.json')


def download(name, url, dest, expected=None, maximum=100_000_000):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        size = dest.stat().st_size
        assert size == expected if expected is not None else 0 < size <= maximum
        progress(name, status='CACHED', bytes=size, expected=expected)
        return dest
    if expected and expected > 1_000_000_000 and urllib.parse.urlsplit(url).hostname == 'download.openmmlab.com':
        return download_segments(name, url, dest, expected)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                n = offset
                with part.open('ab' if offset else 'wb') as stream:
                    while True:
                        block = response.read(4*1024*1024)
                        if not block:
                            break
                        n += len(block)
                        if n > (expected if expected is not None else maximum):
                            raise ValueError('asset exceeds declared bound')
                        stream.write(block)
                        progress(name, status='DOWNLOADING', bytes=n, expected=expected)
            if expected is not None and n != expected:
                raise OSError(f'incomplete asset {n}/{expected}')
            part.replace(dest)
            progress(name, status='DOWNLOADED', bytes=n, expected=expected)
            return dest
        except (OSError, urllib.error.URLError) as exc:
            progress(name, status='RETRY', attempt=attempt, error=type(exc).__name__)
            if attempt == 3:
                raise
            time.sleep((5, 15, 30)[attempt])


def download_segments(name, url, dest, expected):
    """Keep one contiguous prefix and at most four 64 MiB temporary segments."""
    part = dest.with_name(dest.name + '.part')
    chunk = 64 * 1024 * 1024
    for stale in part.parent.glob(part.name + '.seg-*'):
        stale.unlink()

    class SameHost(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if urllib.parse.urlsplit(newurl).hostname != 'download.openmmlab.com':
                raise ValueError('direct checkpoint request changed host')
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    def fetch(start):
        end = min(start + chunk, expected) - 1
        segment = part.with_name(part.name + f'.seg-{start}')
        for attempt in range(4):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), SameHost())
                request = urllib.request.Request(url + f'?segment_start={start}',
                    headers={'Range': f'bytes={start}-{end}'})
                with opener.open(request, timeout=30) as response, segment.open('wb') as output:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{expected}':
                        raise ValueError('checkpoint server returned a different byte range')
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > end-start+1:
                            raise ValueError('segment exceeds declared range')
                        output.write(block)
                if size != end-start+1:
                    raise OSError('incomplete checkpoint segment')
                return segment
            except OSError:
                if attempt == 3:
                    raise
                time.sleep((2, 5, 10)[attempt])

    offset = part.stat().st_size if part.exists() else 0
    assert 0 <= offset <= expected
    progress(name, status='DOWNLOADING_SEGMENTS', bytes=offset, expected=expected,
             connections=4, maximum_segment_staging_bytes=4*chunk)
    with ThreadPoolExecutor(max_workers=4) as pool, part.open('ab') as output:
        while offset < expected:
            starts = range(offset, min(offset+4*chunk, expected), chunk)
            futures = [pool.submit(fetch, start) for start in starts]
            for future in futures:
                segment = future.result()
                size = segment.stat().st_size
                with segment.open('rb') as source:
                    while block := source.read(4*1024*1024):
                        output.write(block)
                output.flush()
                offset += size
                segment.unlink()
                progress(name, status='DOWNLOADING_SEGMENTS', bytes=offset, expected=expected,
                         connections=4, maximum_segment_staging_bytes=4*chunk)
    assert part.stat().st_size == expected
    part.replace(dest)
    progress(name, status='DOWNLOADED', bytes=expected, expected=expected, connections=4)
    return dest


def install_environment():
    progress('environment', status='CREATING')
    envdir = Path(P['environment'])
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(envdir)
    env = dict(os.environ, PATH=str(envdir/'bin')+os.pathsep+os.environ['PATH'])
    packages = [
        'transformers==4.40.1', 'tokenizers==0.19.1', 'huggingface-hub==0.25.2',
        'mmengine==0.10.4', 'mmcv-lite==2.1.0', 'mmpretrain==1.2.0', 'easyocr==1.7.1',
        'model-index==0.1.11', 'modelindex==0.0.2', 'rich==13.7.1', 'addict==2.4.0',
        'yapf==0.40.2', 'termcolor==2.4.0', 'mat4py==0.6.0', 'python-bidi==0.4.2',
        'future==1.0.0', 'pyclipper==1.3.0.post5', 'scikit-image==0.23.2',
        'ninja==1.11.1.1', 'openapi-pydantic==0.4.0', 'func-timeout==4.3.5',
        'makefun==1.15.2', 'langid==1.1.6', 'python-dotenv==1.0.1',
        'pydantic==2.7.1', 'pydantic_core==2.18.2', 'fastapi==0.110.2',
        'starlette==0.37.2', 'imageio==2.34.1', 'lazy_loader==0.4',
        'tifffile==2024.8.30',
        'pycocotools==2.0.7',
        'ordered-set==4.1.0', 'markdown-it-py==3.0.0', 'mdurl==0.1.2',
    ]
    subprocess.run(['python3', '-m', 'pip', 'install', '--no-cache-dir', '--no-deps', *packages],
                   env=env, check=True)
    # Torch 2.8 also exports Adafactor; retain MMEngine's original Transformers optimizer.
    builder = envdir/'lib/python3.10/site-packages/mmengine/optim/optimizer/builder.py'
    old = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)"
    new = "OPTIMIZERS.register_module(name='Adafactor', module=Adafactor, force=True)"
    source = builder.read_text()
    assert source.count(old) == 1 or source.count(new) == 1
    builder.write_text(source.replace(old, new))
    bridge = envdir/'lib/python3.10/site-packages/mmpretrain/models/utils/huggingface.py'
    old_bridge = 'return from_config(cfg)'
    new_bridge = ("return from_config(cfg, **({'attn_implementation': kwargs['attn_implementation']} "
                  "if 'attn_implementation' in kwargs else {}))")
    source = bridge.read_text()
    assert source.count(old_bridge) == 1 or source.count(new_bridge) == 1
    bridge.write_text(source.replace(old_bridge, new_bridge))
    (ROOT/'state/compatibility_patch.json').write_text(json.dumps(dict(
        component='mmengine 0.10.4', patch='allow Transformers Adafactor registration with Torch 2.8',
        model_or_inference_changes=False), indent=2))
    progress('environment', status='INSTALLED')
    return env


def unpack_source(path):
    code = ROOT/'code'
    code.mkdir(exist_ok=True)
    total = 0
    skipped_docs_links = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            if member.issym() and parts[:2] == ('agentlego', 'docs'):
                skipped_docs_links += 1
                continue
            assert not member.issym() and not member.islnk() and '..' not in parts
            dest = code.joinpath(*parts)
            assert dest.resolve().is_relative_to(code.resolve())
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                assert total < 100_000_000
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, dest.open('wb') as out:
                    while block := src.read(1024*1024):
                        out.write(block)
    # The legacy multimodal wrapper cannot negotiate Transformers' automatic SDPA selection.
    benchmark = code/'agentlego/benchmark.py'
    original = 'model=self.model,\n                device=self.device,\n'
    eager = original + "                lang_encoder=dict(attn_implementation='eager'),\n"
    source = benchmark.read_text()
    assert source.count(original) == 3
    benchmark.write_text(source.replace(original, eager))
    (ROOT/'state/attention_compatibility.json').write_text(json.dumps(dict(
        attention_implementation='eager', reason='legacy multimodal wrapper does not advertise SDPA',
        weights_or_architecture_changed=False, training=False), indent=2))
    progress('source', status='EXTRACTED', bytes=total, skipped_docs_links=skipped_docs_links)


def prepare_data():
    import pyarrow.parquet as pq
    from PIL import Image
    table = pq.read_table(ROOT/'input/train.parquet', columns=['image', 'image_path'])
    assert len(table) == 107
    count = 0
    for row in table.to_pylist():
        rel = Path(row['image_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        dest = ROOT/'data'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = row['image']['bytes']
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
        dest.write_bytes(payload)
        count += 1
    assert count == 107
    progress('images', status='READY', files=count, gold_scored=False)


def small_hf_assets():
    from huggingface_hub import snapshot_download
    versions = {}
    for repo in ('liuhaotian/llava-v1.5-7b', 'huggyllama/llama-7b'):
        path = snapshot_download(repo_id=repo, allow_patterns=['*.json', '*.model', 'tokenizer.*'],
                                 revision=P.get('hf_asset_revisions', {}).get(repo),
                                 cache_dir=str(ROOT/'cache/hf/hub'), max_workers=2)
        versions[repo] = Path(path).name
    (ROOT/'state/hf_asset_versions.json').write_text(json.dumps(versions, indent=2))
    progress('tokenizers', status='READY')


def main():
    for rel in ('assets', 'cache/torch/hub/checkpoints', 'cache/easyocr/model', 'data', 'home'):
        (ROOT/rel).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = []
        for asset in P['assets']:
            futures.append((asset, pool.submit(download, asset['name'], asset['url'],
                ROOT/asset['destination'], asset.get('bytes'), asset.get('maximum', 100_000_000))))
        hf = pool.submit(small_hf_assets)
        env = install_environment()
        for asset, future in futures:
            path = future.result()
            if asset['name'] == 'source':
                unpack_source(path)
            elif asset['name'].startswith('ocr_'):
                with zipfile.ZipFile(path) as archive:
                    expected = asset['expanded_name']
                    member = next(x for x in archive.infolist() if Path(x.filename).name == expected)
                    assert member.file_size < 200_000_000
                    with archive.open(member) as src, (ROOT/'cache/easyocr/model'/expected).open('wb') as out:
                        while block := src.read(1024*1024):
                            out.write(block)
        hf.result()
    prepare_data()
    env.update(PYTHONPATH=str(ROOT/'code/agentlego'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    progress('local_validation', status='STARTING')
    available = {}
    for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,noheader,nounits'], text=True).splitlines():
        index, free = (int(item.strip()) for item in line.split(','))
        available[str(index)] = free
    selected = os.environ['TASK_GPU']
    if available.get(selected, 0) < 38000:
        selected = max(available, key=available.get)
    assert available[selected] >= 38000, 'insufficient current GPU headroom for validation'
    env.update(TASK_GPU=selected, CUDA_VISIBLE_DEVICES=selected)
    (ROOT/'state/gpu_choice.json').write_text(json.dumps(dict(
        selected_gpu=selected, free_MiB=available[selected], observed_unix=time.time())))
    env['ENTRY'] = str(ROOT/'src/check_medcta_local_tools.py')
    subprocess.run(['python3', '-u', '-c', 'import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],
                   env=env, cwd=ROOT, check=True)
    result = json.loads((ROOT/'state/local_tools_check.json').read_text())
    assert result['local_tools_passed'] == 4 and result['benchmark_scored'] is False
    (ROOT/'READY.json').write_text(json.dumps(dict(status='READY_FOUR_LOCAL_TOOLS_SEARCH_DEFERRED',
        completed_unix=time.time(), images=107, local_tools=4, search='DEFERRED_NO_KEY',
        training=False, benchmark_scored=False, source_commit=P['source_commit']), indent=2))
    progress('preparation', status='COMPLETE')


if __name__ == '__main__':
    main()
