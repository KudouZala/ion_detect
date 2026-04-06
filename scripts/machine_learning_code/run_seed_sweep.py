import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量运行 SWC-PSWM_plus 多随机种子实验。"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="scripts/machine_learning_code/configs/paper/seeds",
        help="种子配置所在目录",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="SWC-PSWM_plus_seed*.yaml",
        help="用于匹配种子配置的 glob 模式",
    )
    parser.add_argument(
        "--main-script",
        type=str,
        default="scripts/machine_learning_code/main_plus.py",
        help="训练入口脚本路径",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="scripts/machine_learning_code/logs_20260323_2/seeds",
        help="日志输出目录",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若日志文件已存在，则跳过对应 seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不真正运行",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config_dir = (project_root / args.config_dir).resolve()
    main_script = (project_root / args.main_script).resolve()
    log_dir = (project_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg_paths = sorted(config_dir.glob(args.pattern))
    if not cfg_paths:
        raise FileNotFoundError(f"未找到任何配置文件: {config_dir / args.pattern}")

    print(f"📁 config_dir: {config_dir}")
    print(f"🧾 log_dir: {log_dir}")
    print(f"🚀 共发现 {len(cfg_paths)} 个 seed 配置")

    for cfg_path in cfg_paths:
        stem = cfg_path.stem
        log_path = log_dir / f"{stem}.log"
        cmd = [
            sys.executable,
            str(main_script),
            "--config",
            str(cfg_path),
            "--train",
        ]

        print("\n" + "=" * 72)
        print(f"[RUN] {stem}")
        print(" ".join(cmd) + f" > {log_path} 2>&1")

        if args.skip_existing and log_path.exists():
            print(f"[SKIP] 已存在日志: {log_path}")
            continue

        if args.dry_run:
            continue

        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"{stem} 运行失败，退出码: {proc.returncode}，日志: {log_path}")

    print("\n✅ 全部种子任务已完成")


if __name__ == "__main__":
    main()
