import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(square: list[list[int]]) -> bool:
    n = len(square)
    target = n * (n * n + 1) // 2
    if sorted(v for row in square for v in row) != list(range(1, n * n + 1)):
        return False
    if any(sum(row) != target for row in square):
        return False
    if any(sum(square[r][c] for r in range(n)) != target for c in range(n)):
        return False
    if sum(square[i][i] for i in range(n)) != target:
        return False
    if sum(square[i][n - i - 1] for i in range(n)) != target:
        return False
    return True


def test_part1_n3():
    solver = load_module("part1_solver", ROOT / "part1/src/solver.py")
    result = solver.solve_brute_force(3, 60)
    assert result["solution"] is not None
    assert validate(result["solution"])


def test_part2_n3():
    output = ROOT / "tests/.tmp-part2-n3.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "part2/src/solver.py"),
            "--n", "3",
            "--timeout", "60",
            "--workers", "4",
            "--output", str(output),
            "--csv", str(ROOT / "tests/.tmp-part2-n3.csv"),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(output.read_text(encoding="utf-8"))[0]
    output.unlink(missing_ok=True)
    assert result["solution"] is not None
    assert validate(result["solution"])


def test_part2_n2_infeasible():
    output = ROOT / "tests/.tmp-part2-n2.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "part2/src/solver.py"),
            "--n", "2",
            "--timeout", "10",
            "--workers", "2",
            "--output", str(output),
            "--csv", str(ROOT / "tests/.tmp-part2-n2.csv"),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads(output.read_text(encoding="utf-8"))[0]
    output.unlink(missing_ok=True)
    assert result["solution"] is None


def test_part3_n3_and_n2():
    solver = load_module("part3_solver", ROOT / "part3/src/solver.py")
    n3 = solver.solve_cpsat(3, 30, 4)
    n2 = solver.solve_cpsat(2, 5, 1)
    assert n3["solution"] is not None
    assert validate(n3["solution"])
    assert n2["solution"] is None
