#!/usr/bin/env python3
"""Run all tests."""
import sys, os, subprocess

tests_dir = os.path.dirname(os.path.abspath(__file__))
files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]

passed = 0
failed = 0
for f in sorted(files):
    path = os.path.join(tests_dir, f)
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  PASS {f}")
        passed += 1
    else:
        print(f"  FAIL {f}")
        print(result.stdout)
        print(result.stderr)
        failed += 1

print(f"\n{passed}/{len(files)} test files passed")
if failed > 0:
    sys.exit(1)
