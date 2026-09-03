from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Keep CUDA reproducibility and joblib's Windows CPU detection quiet in logs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from scptg.data import load_data
from scptg.metrics import evaluate
from scptg.trainer import train
from scptg.utils import dump_json, make_logger, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scPTG on a labelled .h5ad dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/Zeisel.h5ad"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--pretrain-epochs", type=int, default=None, help="Optional quick-test override")
    parser.add_argument("--joint-epochs", type=int, default=None, help="Optional quick-test override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.pretrain_epochs is not None:
        config["train"]["pretrain_epochs"] = args.pretrain_epochs
    if args.joint_epochs is not None:
        config["train"]["joint_epochs"] = args.joint_epochs
    if args.device is not None:
        config["runtime"]["device"] = args.device

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = make_logger(args.log_dir / f"{args.data.stem}_seed{args.seed}_{stamp}.log")
    started = time.perf_counter()
    set_seed(args.seed, bool(config["runtime"]["deterministic"]))
    device = resolve_device(str(config["runtime"]["device"]))
    logger.info("scPTG started | data=%s | seed=%d | device=%s", args.data, args.seed, device)
    data = load_data(args.data, config["data"])
    logger.info("Data: %s", data.metadata)
    result = train(data.x, data.pca, data.n_clusters, config, device, args.seed, logger)
    metrics = evaluate(data.labels, result.predictions)
    elapsed = time.perf_counter() - started

    run_dir = args.output / data.metadata["dataset"] / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in result.history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(result.history)
    np.savez_compressed(run_dir / "assignments.npz", obs_names=data.obs_names, truth=data.labels,
                        prediction=result.predictions, community=result.community_labels,
                        probabilities=result.probabilities, embedding=result.embedding)
    report = {
        **metrics, "runtime_seconds": elapsed, "seed": args.seed, "device": str(device),
        "trajectory_trust": result.trajectory_trust,
        "trajectory_agreement": result.trajectory_agreement,
        "data": data.metadata, "graph": result.graph.metadata,
    }
    dump_json(report, run_dir / "metrics.json")
    dump_json(config, run_dir / "config.json")
    logger.info("Result | ARI %.4f | NMI %.4f | ACC %.4f", metrics["ari"], metrics["nmi"], metrics["acc"])
    logger.info("Trajectory | trust %.4f | agreement %.4f", result.trajectory_trust, result.trajectory_agreement)
    logger.info("Finished in %.2f s | outputs=%s", elapsed, run_dir.resolve())


if __name__ == "__main__":
    main()
