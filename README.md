# VisualizingPoetry (Poetry-to-Image)

End-to-end framework for poetry-to-image generation with deep semantic understanding, multi-level feature fusion, and poem–image consistency evaluation.

## Environment
- Python 3.10+
- PyTorch / CUDA
- torchvision, timm, transformers
- OpenCLIP / CLIP
- numpy, scipy, scikit-learn
- nltk, spacy
- pillow, opencv-python
- tqdm, pyyaml
- evaluate, bert-score, pycocoevalcap

## Structure
```
code/
  README.md
  requirements.txt
  pyproject.toml
  scripts/
    prepare_data.sh
    train.sh
    evaluate.sh
    infer.sh
  notebooks/
    exploratory.ipynb
  src/
    poetic_vis/
      __init__.py
      configs/
        default.yaml
        dataset.yaml
        model.yaml
        train.yaml
        eval.yaml
      data/
        datasets.py
        datamodule.py
        transforms.py
        text_process.py
      models/
        __init__.py
        pmsum/
          encoder.py
          multitask_heads.py
          exporter.py
        fusion/
          fuse.py
          attention.py
          interaction.py
          decoder.py
        consistency/
          text_relevance.py
          topic_model.py
          clip_similarity.py
          combiner.py
      training/
        losses.py
        optim.py
        schedulers.py
        trainer.py
        logger.py
        checkpoint.py
      evaluation/
        metrics/
          inception_score.py
          fid.py
          clipscore.py
          cider.py
          bertscore.py
          spice.py
        evaluator.py
        reporting.py
      inference/
        pipeline.py
        generate.py
        save_io.py
      utils/
        seed.py
        dist.py
        io.py
        viz.py
      cli/
        train.py
        evaluate.py
        infer.py
    tests/
      test_data.py
      test_models.py
      test_metrics.py
  .gitignore
  LICENSE
  CITATION.cff
```

## Configs
- configs/default.yaml
- configs/dataset.yaml
- configs/model.yaml
- configs/train.yaml
- configs/eval.yaml

## Workflows
- Data preparation: scripts/prepare_data.sh
- Training: python -m poetic_vis.cli.train --config configs/train.yaml
- Evaluation: python -m poetic_vis.cli.evaluate --config configs/eval.yaml
- Inference: python -m poetic_vis.cli.infer --config configs/model.yaml --input x

## Metrics
- IS, FID
- CLIPScore
- CIDEr, BERTScore, SPICE



