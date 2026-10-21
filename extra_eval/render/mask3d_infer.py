#!/usr/bin/env python3
"""Run the official Mask3D ScanNet200 test + export on a scannet200-layout dataset dir.

``main_instance_segmentation.py`` does not run on current PyTorch Lightning (``Trainer``
``weights_save_path`` is gone, and its loggers need ``general.experiment_id``) / PyYAML (``yaml.load``
needs a Loader), and its loss needs detectron2 (shimmed, ``mask3d_shims/``); so this builds
the same thing from Mask3D's own hydra config and code -- ``InstanceSegmentation`` + its
checkpoint loader + ``Trainer.test`` -- without loggers. Everything else (post-processing,
DBSCAN, score sorting, export) is Mask3D's ``trainer/trainer.py``, unmodified. Overrides =
Mask3D's ``scripts/scannet200/scannet200_val.sh`` TEST command (150 queries, top-k 750,
DBSCAN eps 0.95, segment-level masks) + ``general.export=true``.

Run from the Mask3D checkout (``--mask3d_dir``) with a python that has its dependencies; the
export lands in ``<mask3d_dir>/eval_output/instance_evaluation_<name>_0/decoder_-1/``:
``<scene>.txt`` lines ``pred_mask/<scene>_<k>.txt <class> <score>`` in descending score.
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mask3d_dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="scannet200_val.ckpt")
    ap.add_argument("--data_dir", required=True,
                    help="validation_database.yaml (+ train_database.yaml), label_database.yaml, color_mean_std.yaml")
    ap.add_argument("--name", required=True, help="experiment name (export dir)")
    args = ap.parse_args()
    os.chdir(args.mask3d_dir)
    sys.path.insert(0, args.mask3d_dir)
    # detectron2: only Mask3D's loss uses it (see mask3d_shims/detectron2)
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask3d_shims"))
    # Mask3D calls yaml.load(f) without a Loader, which PyYAML >= 6 rejects
    import yaml
    _yaml_load = yaml.load
    yaml.load = lambda stream, Loader=yaml.SafeLoader: _yaml_load(stream, Loader=Loader)
    from hydra.experimental import compose, initialize_config_dir
    from pytorch_lightning import Trainer, seed_everything
    from trainer.trainer import InstanceSegmentation
    from utils.utils import load_checkpoint_with_missing_or_exsessive_keys

    d = args.data_dir
    overrides = [
        f"general.experiment_name={args.name}", "general.project_name=gpt4scene_mask3d",
        f"general.checkpoint={args.checkpoint}", "data/datasets=scannet200", "general.num_targets=201",
        "data.num_labels=200", "general.eval_on_segments=true", "general.train_on_segments=true",
        "general.train_mode=false", "model.num_queries=150", "general.topk_per_image=750",
        "general.use_dbscan=true", "general.dbscan_eps=0.95", "general.export=true", "logging=minimal",
        f"general.save_dir={d}/saved_{args.name}",
    ]
    for split in ["train", "validation", "test"]:
        overrides += [f"data.{split}_dataset.data_dir={d}",
                      f"data.{split}_dataset.label_db_filepath={d}/label_database.yaml",
                      f"data.{split}_dataset.color_mean_std={d}/color_mean_std.yaml"]
    with initialize_config_dir(config_dir=os.path.join(args.mask3d_dir, "conf")):
        cfg = compose(config_name="config_base_instance_segmentation.yaml", overrides=overrides)
    seed_everything(cfg.general.seed)
    model = InstanceSegmentation(cfg)
    cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    Trainer(accelerator="gpu", devices=1, logger=False).test(model)


if __name__ == "__main__":
    main()
