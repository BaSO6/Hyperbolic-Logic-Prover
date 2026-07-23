from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "index",
        "name",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_pct",
        "power_draw_w",
    ]
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        while not args.stop_file.exists():
            command = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            try:
                output = subprocess.check_output(command, text=True)
                for line in output.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) == 6:
                        writer.writerow(
                            dict(
                                zip(
                                    fields,
                                    [datetime.now(timezone.utc).isoformat(), *values],
                                )
                            )
                        )
                handle.flush()
            except Exception:
                pass
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
