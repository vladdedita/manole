"""Tests for benchmark_extractors — benchmark script comparing docling vs kreuzberg."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- Acceptance test (step 03-01) ---


def test_benchmark_runs_both_backends_and_reports_timing_and_lengths():
    """Acceptance: benchmark runs warmup + 3 trials per backend per file,
    reports mean times and output lengths for parity comparison."""
    from benchmark_extractors import run_benchmark

    # Create a temp directory with two test files
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "doc1.pdf").write_bytes(b"fake pdf 1")
        (Path(tmpdir) / "doc2.pdf").write_bytes(b"fake pdf 2")

        # Mock FileReader.from_backend to return readers with predictable behavior
        mock_readers = {}

        def fake_from_backend(backend, **kwargs):
            reader = MagicMock()
            call_count = {"n": 0}

            def fake_read(path):
                call_count["n"] += 1
                return f"extracted by {backend} from {Path(path).name} (call {call_count['n']})"

            reader.read.side_effect = fake_read
            mock_readers[backend] = reader
            return reader

        with patch("benchmark_extractors.FileReader") as mock_file_reader:
            mock_file_reader.from_backend.side_effect = fake_from_backend
            results = run_benchmark(tmpdir, num_trials=3)

        # Verify structure: results for both backends
        assert "docling" in results
        assert "kreuzberg" in results

        for backend in ("docling", "kreuzberg"):
            backend_results = results[backend]
            # Should have results for each file
            assert len(backend_results) == 2

            for file_result in backend_results:
                # Each result has: filename, mean_time, output_length
                assert "filename" in file_result
                assert "mean_time" in file_result
                assert "output_length" in file_result
                # mean_time is a positive float (from 3 trials)
                assert isinstance(file_result["mean_time"], float)
                assert file_result["mean_time"] >= 0
                # output_length is a positive int
                assert isinstance(file_result["output_length"], int)
                assert file_result["output_length"] > 0

        # Verify each backend reader was called:
        # 1 warmup + (3 trials x 2 files) = 7 calls per backend
        for backend in ("docling", "kreuzberg"):
            assert mock_readers[backend].read.call_count == 7


# --- Unit tests (step 03-01) ---
# Test Budget: 4 behaviors x 2 = 8 unit tests max. Using 4.


def test_benchmark_computes_mean_time_from_trials():
    """run_benchmark() computes mean wall-clock time across num_trials for each file."""
    from benchmark_extractors import run_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "single.pdf").write_bytes(b"pdf")

        def fake_from_backend(backend, **kwargs):
            reader = MagicMock()
            reader.read.return_value = "extracted text"
            return reader

        with patch("benchmark_extractors.FileReader") as mock_fr:
            mock_fr.from_backend.side_effect = fake_from_backend
            results = run_benchmark(tmpdir, num_trials=5)

        # Mean time must be a float (we can't predict exact value but it must be >= 0)
        for backend in ("docling", "kreuzberg"):
            assert len(results[backend]) == 1
            assert results[backend][0]["mean_time"] >= 0.0
            assert isinstance(results[backend][0]["mean_time"], float)


def test_benchmark_reports_output_length_per_backend():
    """run_benchmark() reports output length (chars) for each backend-file pair."""
    from benchmark_extractors import run_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.pdf").write_bytes(b"pdf")

        def fake_from_backend(backend, **kwargs):
            reader = MagicMock()
            if backend == "docling":
                reader.read.return_value = "short"
            else:
                reader.read.return_value = "a much longer text output"
            return reader

        with patch("benchmark_extractors.FileReader") as mock_fr:
            mock_fr.from_backend.side_effect = fake_from_backend
            results = run_benchmark(tmpdir, num_trials=1)

        assert results["docling"][0]["output_length"] == len("short")
        assert results["kreuzberg"][0]["output_length"] == len("a much longer text output")


def test_benchmark_handles_backend_error_gracefully():
    """run_benchmark() records error and continues when a backend fails on a file."""
    from benchmark_extractors import run_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "bad.pdf").write_bytes(b"pdf")
        (Path(tmpdir) / "good.pdf").write_bytes(b"pdf")

        def fake_from_backend(backend, **kwargs):
            reader = MagicMock()
            if backend == "docling":
                def read_with_error(path):
                    if "bad" in str(path):
                        raise RuntimeError("conversion failed")
                    return "good result"
                reader.read.side_effect = read_with_error
            else:
                reader.read.return_value = "kreuzberg output"
            return reader

        with patch("benchmark_extractors.FileReader") as mock_fr:
            mock_fr.from_backend.side_effect = fake_from_backend
            results = run_benchmark(tmpdir, num_trials=1)

        # Docling should have error for bad.pdf but success for good.pdf
        docling_results = {r["filename"]: r for r in results["docling"]}
        assert "error" in docling_results["bad.pdf"]
        assert docling_results["good.pdf"]["output_length"] > 0

        # Kreuzberg should succeed on both
        assert len(results["kreuzberg"]) == 2
        assert all("error" not in r for r in results["kreuzberg"])


def test_benchmark_cli_parses_dir_argument():
    """CLI entry point parses --dir argument and calls run_benchmark."""
    from benchmark_extractors import build_parser

    parser = build_parser()
    args = parser.parse_args(["--dir", "/some/path"])
    assert args.dir == "/some/path"
