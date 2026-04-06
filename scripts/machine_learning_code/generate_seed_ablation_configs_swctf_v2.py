from pathlib import Path

import yaml


SEEDS = [123, 2025, 3407, 27182, 31415]
VARIANTS = [
    "SWC-PSWM_plus_swctf",
    "exp_a2_plus_swctf",
    "exp_b2_plus_swctf",
    "exp_c2_plus_swctf",
    "exp_d2_plus_swctf",
]
SEED_NUM_EPOCHS = 1000


def main():
    cfg_root = Path(__file__).resolve().parent / "configs" / "paper"
    out_dir = cfg_root / "seeds_swctf_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        template_path = cfg_root / f"{variant}.yaml"
        with open(template_path, "r", encoding="utf-8") as f:
            template = yaml.safe_load(f)

        for seed in SEEDS:
            cfg = yaml.safe_load(
                yaml.safe_dump(template, sort_keys=False, allow_unicode=True)
            )
            cfg["experiment"]["name"] = f"{variant}_seed{seed}"
            cfg["experiment"]["seed"] = int(seed)
            cfg["train"]["num_epochs"] = int(SEED_NUM_EPOCHS)

            if variant == "SWC-PSWM_plus_swctf":
                cfg["paths"]["data_folder"] = "datasets/datasets_for_all_plus"
                cfg["paths"]["test_folder"] = "datasets/datasets_for_all_test_no_overlap"
            else:
                split_name = f"SWC-PSWM_plus_swctf_seed{seed}"
                cfg["paths"]["data_folder"] = (
                    f"datasets/datasets_for_all_train_no_overlap/{split_name}"
                )
                cfg["paths"]["test_folder"] = (
                    f"datasets/datasets_for_all_test_no_overlap/{split_name}"
                )
            # 并行安全：每个 seed 使用独立统计量文件，避免并发覆盖
            cfg["paths"]["stats_file"] = f"datasets/stats/stats_dataset_swctf_seed{seed}.json"

            out_path = out_dir / f"{variant}_seed{seed}.yaml"
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
