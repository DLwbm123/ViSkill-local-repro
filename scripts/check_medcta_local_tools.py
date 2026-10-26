"""Functional smoke checks on synthetic inputs, without benchmark scoring."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(os.environ['TASK_ROOT'])
passed = []


def mark_passed(name):
    passed.append(name)
    (ROOT/'state/local_tools_progress.json').write_text(json.dumps(dict(
        passed_tools=passed, updated_unix=time.time(), benchmark_scored=False)))


assert torch.cuda.is_available()
spec = importlib.util.spec_from_file_location('local_tool_definitions', ROOT/'code/agentlego/benchmark.py')
definitions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(definitions)
from agentlego.tools import Calculator, OCR
from agentlego.types import ImageIO

image = Image.new('RGB', (640, 320), 'white')
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
draw.text((30, 25), 'MEDICAL 123', font=font, fill='black')
draw.rectangle((50, 140, 220, 280), fill='red')
fixture = ROOT/'state/synthetic_fixture.png'
image.save(fixture)
payload = ImageIO(str(fixture))

calculator = Calculator()
assert str(calculator.apply('2 + 3 * 4')) == '14'
mark_passed('Calculator')
ocr = OCR(device='cuda')
ocr.setup()
text = ocr.apply(payload)
assert '123' in text, 'OCR failed to read the synthetic digit fixture'
mark_passed('OCR')

description = definitions.ImageDescription(device='cuda')
description.setup()
region = definitions.RegionAttributeDescription(device='cuda')
region.setup()
assert description._inferencer is region._inferencer, 'expected identical model calls to share the cached inferencer'
whole = description.apply(payload)
assert isinstance(whole, str) and whole.strip()
mark_passed('ImageDescription')
local = region.apply(payload, '(40, 130, 240, 300)', 'color and shape')
assert isinstance(local, str) and local.strip()
mark_passed('RegionAttributeDescription')
versions = {}
for name in ('torch', 'torchvision', 'transformers', 'mmengine', 'mmcv-lite', 'mmpretrain', 'easyocr'):
    versions[name] = importlib.metadata.version(name)
result = dict(status='LOCAL_TOOLS_FUNCTIONAL', completed_unix=time.time(), local_tools_passed=4,
    calculator=True, ocr=True, image_description=True, region_attribute_description=True,
    shared_visual_model=True, synthetic_generation_calls=2, benchmark_scored=False,
    serper_called=False, medical_gold_read=False, versions=versions,
    device=torch.cuda.get_device_name(0), outputs=dict(ocr_characters=len(text),
        description_characters=len(whole), region_characters=len(local)))
(ROOT/'state/local_tools_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
