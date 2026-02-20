"""Benchmark script comparing docling vs kreuzberg text extraction backends.

Usage: python benchmark_extractors.py --dir /path/to/documents
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from file_reader import FileReader


BACKENDS = ("docling", "kreuzberg")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".htm"}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark docling vs kreuzberg text extraction"
    )
    parser.add_argument(
        "--dir", required=True, help="Directory containing documents to benchmark"
    )
    parser.add_argument(
        "--trials", type=int, default=3, help="Number of timed trials per file (default: 3)"
    )
    return parser


def _collect_files(directory: str) -> list[Path]:
    """Collect supported document files from directory, sorted by name."""
    dir_path = Path(directory)
    files = [
        f for f in sorted(dir_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def run_benchmark(
    directory: str, num_trials: int = 3
) -> dict[str, list[dict]]:
    """Run benchmark on all supported files in directory with both backends.

    For each backend:
      1. Warmup: extract first file once (untimed)
      2. Run num_trials timed extractions per file
      3. Report mean time and output length per file

    Returns dict mapping backend name to list of per-file result dicts.
    Each result dict has: filename, mean_time, output_length (or error on failure).
    """
    files = _collect_files(directory)
    results: dict[str, list[dict]] = {}

    for backend in BACKENDS:
        reader = FileReader.from_backend(backend)
        backend_results: list[dict] = []

        # Warmup: run one extraction on first file (untimed)
        if files:
            try:
                reader.read(str(files[0]))
            except Exception:
                pass  # warmup errors are ignored

        # Timed trials
        for file_path in files:
            try:
                times: list[float] = []
                output = ""
                for _ in range(num_trials):
                    start = time.perf_counter()
                    output = reader.read(str(file_path))
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)

                mean_time = sum(times) / len(times)
                backend_results.append({
                    "filename": file_path.name,
                    "mean_time": mean_time,
                    "output_length": len(output),
                })
            except Exception as e:
                backend_results.append({
                    "filename": file_path.name,
                    "error": str(e),
                })

        results[backend] = backend_results

    return results


def format_results(results: dict[str, list[dict]]) -> str:
    """Format benchmark results as a readable table."""
    lines: list[str] = []
    header = f"{'File':<30} {'Backend':<12} {'Mean Time (s)':<15} {'Output Len':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    # Collect all filenames
    all_files: set[str] = set()
    for backend_results in results.values():
        for r in backend_results:
            all_files.add(r["filename"])

    for filename in sorted(all_files):
        for backend in BACKENDS:
            backend_results = results.get(backend, [])
            file_result = next((r for r in backend_results if r["filename"] == filename), None)
            if file_result is None:
                continue
            if "error" in file_result:
                lines.append(f"{filename:<30} {backend:<12} {'ERROR':<15} {file_result['error']}")
            else:
                lines.append(
                    f"{filename:<30} {backend:<12} {file_result['mean_time']:<15.4f} {file_result['output_length']:<12}"
                )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args.dir, num_trials=args.trials)
    print(format_results(results))


if __name__ == "__main__":
    main()
