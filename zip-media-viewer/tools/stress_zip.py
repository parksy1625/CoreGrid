#!/usr/bin/env python3
"""Deterministic ZIP central-directory and random-access stress smoke test.
This tests Python ZIP fixtures, not Android device memory or UI performance.
"""
import argparse
import hashlib
import os
import random
import tempfile
import time
import zipfile

def run(count, seed):
    rng = random.Random(seed)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stress.zip")
        expected = {}
        with zipfile.ZipFile(path, "w", allowZip64=True) as z:
            for i in range(count):
                name = f"album_{i % 37:02d}/img_{i:08d}.jpg"
                payload = hashlib.sha256(f"{seed}-{i}".encode()).digest()
                z.writestr(name, payload, compress_type=zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED)
                if i % max(1, count // 500) == 0:
                    expected[name] = payload
        with zipfile.ZipFile(path) as z:
            entries = z.infolist()
            assert len(entries) == count, (len(entries), count)
            assert len(set(e.filename for e in entries)) == count
            samples = rng.sample(entries, min(500, len(entries)))
            for e in samples:
                assert z.read(e.filename) == hashlib.sha256(f"{seed}-{int(e.filename.rsplit('_', 1)[1].split('.')[0])}".encode()).digest()
            for name, payload in expected.items():
                assert z.read(name) == payload
        elapsed = time.perf_counter() - start
        print(f"PASS entries={count} bytes={os.path.getsize(path)} seconds={elapsed:.2f}")
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entries", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.entries, args.seed)
