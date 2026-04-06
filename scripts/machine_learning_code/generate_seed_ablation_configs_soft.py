from pathlib import Path

import yaml


SEEDS = [123, 2025, 3407, 27182, 31415]
VARIANTS = [
    "SWC-PSWM_plus",
    "exp_a_plus",
    "exp_b_plus",
    "exp_c_plus",
    "exp_d_plus",
]
SEED_NUM_EPOCHS = 600
OUT_SUBDIR = "seeds_soft_v1"

# Soft-constraint enhanced profile (moderate boost).
SOFT_PROFILE = {
    "trainer": {
        "gamma": 2.0e-4,
        "lambda_group": 0.002,
        "lambda_tree": 0.002,
    },
    "train_pairs": {
        "lambda_consistency": 0.002,
        "lambda_monodec": 0.002,
        "lambda_polarity": 0.001,
    },
}


def _deep_copy_cfg(cfg: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def _apply_soft_profile(cfg: dict, variant: str):
    tr = cfg["train"]["trainer"]
    tp = cfg["train"]["train_pairs"]
    tr["gamma"] = float(SOFT_PROFILE["trainer"]["gamma"])
    tr["lambda_group"] = float(SOFT_PROFILE["trainer"]["lambda_group"])
    tr["lambda_tree"] = float(SOFT_PROFILE["trainer"]["lambda_tree"])
    # Keep Exp-B as the strict "w/o pair consistency" ablation.
    if variant != "exp_b_plus":
        tp["lambda_consistency"] = float(SOFT_PROFILE["train_pairs"]["lambda_consistency"])
    tp["lambda_monodec"] = float(SOFT_PROFILE["train_pairs"]["lambda_monodec"])
    tp["lambda_polarity"] = float(SOFT_PROFILE["train_pairs"]["lambda_polarity"])


def main():
    cfg_root = Path(__file__).resolve().parent / "configs" / "paper"
    out_dir = cfg_root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        template_path = cfg_root / f"{variant}.yaml"
        with open(template_path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f)

        for seed in SEEDS:
            cfg = _deep_copy_cfg(template)
            cfg["experiment"]["name"] = f"{variant}_soft_v1_seed{seed}"
            cfg["experiment"]["seed"] = int(seed)
            cfg["train"]["num_epochs"] = int(SEED_NUM_EPOCHS)

            _apply_soft_profile(cfg, variant)

            if variant == "SWC-PSWM_plus":
                cfg["paths"]["data_folder"] = "datasets/datasets_for_all_plus"
                cfg["paths"]["test_folder"] = "datasets/datasets_for_all_test_no_overlap"
            else:
                split_name = f"SWC-PSWM_plus_soft_v1_seed{seed}"
                cfg["paths"]["data_folder"] = (
                    f"datasets/datasets_for_all_train_no_overlap/{split_name}"
                )
                cfg["paths"]["test_folder"] = (
                    f"datasets/datasets_for_all_test_no_overlap/{split_name}"
                )

            out_path = out_dir / f"{variant}_soft_v1_seed{seed}.yaml"
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
