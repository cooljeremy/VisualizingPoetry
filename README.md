# Poetry-to-Image (PoeticVis)

End-to-end framework for poetry-to-image generation with deep semantic understanding, multi-level feature fusion, and poem–image consistency evaluation. Implements the three core modules described in the paper:
- PMSUM: pre-trained text encoder + multi-task heads (emotion/imagery/rhetoric)
- E‑I‑R FFGM: multi-level fusion (f_fuse1/g1, f_fuse2/g2), attention aggregation, tensor interaction, Transformer-based prompt decoder
- P‑I CEM: textual relevance (semantic + topic), image–text similarity (OpenCLIP), combined score with thresholded regeneration loop

## Environment
- Python 3.10+
- CUDA-capable PyTorch recommended
- Install dependencies:
  - pip install -r code/requirements.txt


## Data format
- Splits are JSONL files under `data/` as referenced by `configs/dataset.yaml`
- Each line is a JSON object with fields:
  - poem: string
  - image: relative path to image file
  - emo: int label (optional)
  - img: int label (optional)
  - rhe: int label (optional)
  - tgt_ids: list of ints for decoder target sequence (optional; PAD=0)

Example line:
```
{"poem":"床前明月光...", "image":"images/0001.jpg", "emo":2, "img":5, "rhe":3, "tgt_ids":[101, 2009, 102]}
```

## Configs
- code/src/poetic_vis/configs/default.yaml
- code/src/poetic_vis/configs/dataset.yaml
- code/src/poetic_vis/configs/model.yaml
- code/src/poetic_vis/configs/train.yaml
- code/src/poetic_vis/configs/eval.yaml

## Quick start
Prepare directories:
```
bash code/scripts/prepare_data.sh
```

Train (uses PoeticModel):
```
bash code/scripts/train.sh
# or
python -m poetic_vis.cli.train --config code/src/poetic_vis/configs/train.yaml
```

Inference (single poem):
```
bash code/scripts/infer.sh
# or
python -m poetic_vis.cli.infer --config code/src/poetic_vis/configs/model.yaml --input "床前明月光，疑是地上霜。"
```

## Consistency evaluation (P‑I CEM)
Compute S_textual, S_image-text, S_total with an optional regeneration loop:
```python
from PIL import Image
from poetic_vis.models.consistency.pipeline import evaluate_and_maybe_regenerate

def regenerate_fn(prompts):
    # return (new_prompts, new_images) of same length
    return prompts, [Image.open("path.jpg") for _ in prompts]

poems = ["床前明月光..."]
prompts = ["a bright moon ..."]
images = [Image.open("path.jpg")]
res = evaluate_and_maybe_regenerate(poems, prompts, images, alpha=0.5, beta=0.5, thresh=0.8, regenerate_fn=regenerate_fn, max_iter=2, device="cuda")
print(res)
```

## Notes
- IS/FID: require GPU and InceptionV3; images accepted as [B,3,H,W] tensors in [0,1] or [0,255]
- CLIPScore: uses OpenCLIP (ViT-B/32 by default); inputs are PIL images and texts
- CIDEr/SPICE: require pycocoevalcap; inputs are dicts {image_id: [caption]}
- BERTScore: set `lang="zh"` for Chinese or appropriate language code for English



