"""
run_lens_queue.py -- run a list of lens_train_diag.py jobs with at most
--slots running at once (RAM is the limit on a 16 GB laptop: each training
process needs ~0.6-0.8 GB; more processes than RAM allows -> paging and a
5-10x slowdown, measured 2026-10-19).

Job file: one job per line, "<run_name> <lens_train_diag.py args...>"; '#' = comment.
Example line:
  cv_base_s0 --model cv_fft --seed 0 --epochs 150 --out results/lens_diag/main

Run from the fire198 root:
  python cvnn/GravLens_Codes/run_lens_queue.py jobs.txt --slots 3 --threads 3
Jobs whose <out>/<run_name>/summary.json already exists are skipped, so an
interrupted queue can simply be restarted.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--logdir", default="results/lens_diag/logs")
    a = ap.parse_args()
    os.makedirs(a.logdir, exist_ok=True)

    jobs = []
    for line in open(a.jobfile):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, *args = shlex.split(line)
        out = args[args.index("--out") + 1] if "--out" in args else "results/lens_diag"
        if os.path.exists(os.path.join(out, name, "summary.json")):
            print(f"skip {name} (done)", flush=True)
            continue
        jobs.append((name, args))

    running = []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < a.slots:
            name, args = jobs.pop(0)
            log = open(os.path.join(a.logdir, name + ".log"), "w")
            cmd = [sys.executable, os.path.join(HERE, "lens_train_diag.py"), *args,
                   "--run-name", name, "--threads", str(a.threads)]
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[{(time.time()-t0)/60:6.1f} min] start {name}", flush=True)
        for item in list(running):
            name, p, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                print(f"[{(time.time()-t0)/60:6.1f} min] end   {name} rc={p.returncode}", flush=True)
        time.sleep(5)
    print(f"all done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
