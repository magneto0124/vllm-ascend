# SPDX-License-Identifier: Apache-2.0
"""Run the first test.py input with bounded graph metadata diagnostics."""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="test.py")
    args = parser.parse_args()
    namespace = runpy.run_path(str(Path(args.test).resolve()))
    config = namespace["main"].__globals__
    config.update(num_prompts=1, enable_profiling=False, enforce_eager=False)
    config["results_dir"] = str(Path(config["results_dir"]) / "graph_diagnostic")
    print("GDN audit: first input, FULL requested, profiler off; report [GDN-AUDIT] lines", flush=True)
    config["main"]()


if __name__ == "__main__":
    main()
