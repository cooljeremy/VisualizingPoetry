import argparse
import yaml
from pathlib import Path
import torch
from poetic_vis.utils.seed import set_seed
from poetic_vis.utils.dist import device_of
from poetic_vis.data.datamodule import build_dataloaders
from poetic_vis.training.trainer import Trainer
from poetic_vis.training.optim import build_optimizer
from poetic_vis.training.schedulers import build_scheduler
from poetic_vis.models.poetic_model import PoeticModel


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_step_fn(model_cfg):
    ce = torch.nn.CrossEntropyLoss()
    dec_ce = torch.nn.CrossEntropyLoss(ignore_index=0)

    def step_fn(model, batch, device: str):
        texts = batch.get("poem", [])
        if isinstance(texts, torch.Tensor):
            texts = [t for t in texts]
        tgt_ids = batch.get("tgt_ids", None)
        if tgt_ids is not None and isinstance(tgt_ids, list):
            max_len = max(len(x) for x in tgt_ids) if len(tgt_ids) > 0 else 0
            if max_len > 0:
                padded = torch.zeros((len(tgt_ids), max_len), dtype=torch.long)
                for i, seq in enumerate(tgt_ids):
                    padded[i, :len(seq)] = torch.tensor(seq[:max_len], dtype=torch.long)
                tgt_ids = padded
            else:
                tgt_ids = None
        if tgt_ids is not None and isinstance(tgt_ids, torch.Tensor):
            tgt_ids = tgt_ids.to(device)
        out = model(texts, tgt_ids=tgt_ids)
        heads = out["heads"]
        B = heads["emo"].shape[0]
        emo_t = batch.get("emo", None)
        img_t = batch.get("img", None)
        rhe_t = batch.get("rhe", None)
        if emo_t is None:
            emo_t = torch.zeros(B, dtype=torch.long)
        if img_t is None:
            img_t = torch.zeros(B, dtype=torch.long)
        if rhe_t is None:
            rhe_t = torch.zeros(B, dtype=torch.long)
        emo_t = emo_t.to(device)
        img_t = img_t.to(device)
        rhe_t = rhe_t.to(device)
        loss_mtl = ce(heads["emo"], emo_t) + ce(heads["img"], img_t) + ce(heads["rhe"], rhe_t)
        loss_dec = torch.tensor(0.0, device=device)
        if "logits" in out:
            logits = out["logits"]
            if tgt_ids is not None and logits.dim() == 3:
                loss_dec = dec_ce(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
        return loss_mtl + loss_dec

    return step_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = device_of(cfg.get("device", "cuda"))
    dsc = load_yaml(str(Path(args.config).parent / "dataset.yaml"))
    mdl = load_yaml(str(Path(args.config).parent / "model.yaml"))
    train_loader, val_loader, _ = build_dataloaders(dsc["data_root"], dsc["train_split"], dsc["val_split"], dsc["test_split"], dsc["image_size"], dsc["batch_size"], dsc["num_workers"], dsc["max_length"])
    model = PoeticModel(text_model=mdl["text_model"], hidden_dim=mdl["hidden_dim"], fusion_dim=mdl["fusion_dim"], decoder_layers=mdl["decoder_layers"], num_heads=mdl["num_heads"])
    opt = build_optimizer(model.parameters(), cfg["lr"], cfg["weight_decay"])
    sch = build_scheduler(opt, cfg["epochs"])
    step_fn = build_step_fn(mdl)
    trainer = Trainer(model, opt, sch, step_fn, str(device), "outputs", cfg.get("log_interval", 50))
    epochs = cfg["epochs"]
    trainer.fit(train_loader, val_loader, epochs)


if __name__ == "__main__":
    main()


