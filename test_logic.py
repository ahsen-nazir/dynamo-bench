"""Tests for benchmark requests, trace metrics, and dashboard output."""

import io
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from dashboard import LatencyChart, build_dashboard
from logic import send_request
from main import (
    _active_request_seconds,
    _label_worker_id,
    _prefix_cache_metrics,
    _prefix_hit_rate,
    _update_worker_metrics,
    latest_request_metrics,
    _record_prompt_workers,
    _trace_prefix_for_experiment,
)
from prompts import PROMPT_FAMILIES, SHARED_PROMPT, build_requests


class BenchmarkTests(unittest.TestCase):
    def test_trace_prefix_matches_selected_router(self):
        self.assertEqual(_trace_prefix_for_experiment("round-robin"), "./traces/rr")
        self.assertEqual(_trace_prefix_for_experiment("kv-router"), "./traces/kv")
        self.assertIsNone(_trace_prefix_for_experiment("cache-reuse"))

    def test_active_request_time_excludes_gaps_and_merges_overlap(self):
        results = [
            {"_request_started_at": 1.0, "_response_received_at": 2.0},
            {"_request_started_at": 4.0, "_response_received_at": 5.5},
            {"_request_started_at": 1.5, "_response_received_at": 3.0},
        ]

        self.assertEqual(_active_request_seconds(results), 3.5)

    def test_worker_labels_increment_only_for_new_ids(self):
        worker_labels = {}

        self.assertEqual(_label_worker_id(123456789, worker_labels), "w0")
        self.assertEqual(_label_worker_id(987654321, worker_labels), "w1")
        self.assertEqual(_label_worker_id(123456789, worker_labels), "w0")
        self.assertEqual(_label_worker_id("", worker_labels), "")
        self.assertEqual(len(worker_labels), 2)

    def test_prefix_cache_scrape_extracts_only_the_requested_metrics(self):
        response = io.BytesIO(
            b"other_metric 99\n"
            b'vllm:prefix_cache_hits_total{worker="w0"} 12\n'
            b"vllm:prefix_cache_queries_total 20\n"
        )
        with patch("main.urllib.request.urlopen", return_value=response) as urlopen:
            self.assertEqual(
                _prefix_cache_metrics(8100),
                {
                    "vllm:prefix_cache_hits_total": 12.0,
                    "vllm:prefix_cache_queries_total": 20.0,
                },
            )

        urlopen.assert_called_once_with(
            "http://localhost:8100/metrics", timeout=2
        )

    def test_prefix_cache_scrape_failure_returns_no_value(self):
        with (
            patch("main.urllib.request.urlopen", side_effect=OSError("offline")),
            patch("main.console.print"),
            patch("main._debug_log"),
        ):
            self.assertIsNone(_prefix_cache_metrics(8110))

    def test_prefix_hit_rate_is_a_percentage_and_handles_zero_queries(self):
        self.assertEqual(_prefix_hit_rate(12, 20), 60.0)
        self.assertIsNone(_prefix_hit_rate(0, 0))

    def test_request_reads_cached_tokens_from_its_vllm_usage_chunk(self):
        response = io.BytesIO(
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens_details":'
            b'{"cached_tokens":64}}}\n\n'
            b"data: [DONE]\n\n"
        )
        with patch("logic.urllib.request.urlopen", return_value=response) as urlopen:
            measurement = send_request("prompt", request_id="R7")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("X-request-id"), "R7")
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertEqual(measurement["cached_tokens"], 64)

    def test_worker_prefix_hit_rate_uses_independent_counter_deltas(self):
        worker_metrics = {
            worker: {"hits": None, "queries": None, "prefix_hit_rate": ""}
            for worker in ("w0", "w1")
        }
        w0_metrics = {
            "vllm:prefix_cache_hits_total": 10,
            "vllm:prefix_cache_queries_total": 20,
        }
        w1_metrics = {
            "vllm:prefix_cache_hits_total": 8,
            "vllm:prefix_cache_queries_total": 10,
        }

        self.assertEqual(
            _update_worker_metrics(worker_metrics, "w0", w0_metrics)[
                "prefix_hit_rate"
            ],
            "",
        )
        self.assertEqual(
            _update_worker_metrics(worker_metrics, "w1", w1_metrics)[
                "prefix_hit_rate"
            ],
            "",
        )
        self.assertEqual(
            _update_worker_metrics(
                worker_metrics,
                "w0",
                {
                    "vllm:prefix_cache_hits_total": 13,
                    "vllm:prefix_cache_queries_total": 30,
                },
            )["prefix_hit_rate"],
            "30.00",
        )
        self.assertEqual(worker_metrics["w1"]["hits"], 8)
        self.assertEqual(worker_metrics["w1"]["queries"], 10)

    def test_worker_prefix_hit_rate_resets_baseline_when_counters_decrease(self):
        worker_metrics = {
            "w0": {"hits": 10, "queries": 20, "prefix_hit_rate": "50.00"},
        }
        snapshot = _update_worker_metrics(
            worker_metrics,
            "w0",
            {
                "vllm:prefix_cache_hits_total": 2,
                "vllm:prefix_cache_queries_total": 5,
            },
        )

        self.assertEqual(snapshot["hits"], 2)
        self.assertEqual(snapshot["queries"], 5)
        self.assertEqual(snapshot["prefix_hit_rate"], "")

    def test_worker_metric_diagnostic_logs_previous_current_and_delta(self):
        worker_metrics = {
            "w0": {"hits": 10, "queries": 20, "prefix_hit_rate": ""},
        }
        with patch("main._debug_log") as debug_log:
            _update_worker_metrics(
                worker_metrics,
                "w0",
                {
                    "vllm:prefix_cache_hits_total": 13,
                    "vllm:prefix_cache_queries_total": 30,
                },
                "R2",
            )

        diagnostic = debug_log.call_args.args[0]
        self.assertIn("request=R2", diagnostic)
        self.assertIn("actual_prefill_worker=w0", diagnostic)
        self.assertIn("previous_hits=10", diagnostic)
        self.assertIn("previous_queries=20", diagnostic)
        self.assertIn("current_hits=13", diagnostic)
        self.assertIn("current_queries=30", diagnostic)
        self.assertIn("delta_hits=3", diagnostic)
        self.assertIn("delta_queries=10", diagnostic)
        self.assertIn("calculated_rate=30.00%", diagnostic)

    def test_cache_column_reports_prior_prefill_and_decode_workers(self):
        worker_labels = {}
        prompt_workers = {}

        self.assertEqual(
            _record_prompt_workers(
                "prompt-a", 123, 456, worker_labels, prompt_workers
            ),
            ("w0", "w1", "-"),
        )
        self.assertEqual(
            _record_prompt_workers(
                "prompt-a", 789, 456, worker_labels, prompt_workers
            ),
            ("w2", "w1", "w0"),
        )
        self.assertEqual(
            _record_prompt_workers(
                "prompt-b", 456, 789, worker_labels, prompt_workers
            ),
            ("w1", "w2", "-"),
        )

    def test_latest_trace_reader_returns_workers_only(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            old_path = Path(f"{prefix}.9.jsonl.gz")
            latest_path = Path(f"{prefix}.10.jsonl.gz")
            with gzip.open(old_path, "wt", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "event": {
                                "event_type": "request_end",
                                "request": {
                                    "worker": {
                                        "prefill_worker_id": 9,
                                        "decode_worker_id": 9,
                                    }
                                },
                            }
                        }
                    )
                    + "\n"
                )
            with gzip.open(latest_path, "wt", encoding="utf-8") as trace:
                trace.write("{invalid json}\n")
                trace.write(
                    json.dumps(
                        {
                            "event": {
                                "event_type": "request_end",
                                "request": {
                                    "worker": {
                                        "prefill_worker_id": 10,
                                        "decode_worker_id": 11,
                                    },
                                    "cached_tokens": 999,
                                },
                            }
                        }
                    )
                    + "\n"
                )
                trace.write(
                    json.dumps(
                        {"event": {"event_type": "router_selection"}}
                    )
                    + "\n"
                )

            self.assertEqual(
                latest_request_metrics(str(prefix), timeout=0),
                (10, 11),
            )

    def test_latest_trace_reader_matches_request_received_time_window(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            with gzip.open(f"{prefix}.1.jsonl.gz", "wt", encoding="utf-8") as trace:
                for received_ms, worker_id in (
                    (1000, 10),
                    (1500, 20),
                    (2000, 30),
                ):
                    trace.write(
                        json.dumps(
                            {
                                "event": {
                                    "event_type": "request_end",
                                    "request": {
                                        "request_received_ms": received_ms,
                                        "worker": {
                                            "prefill_worker_id": worker_id,
                                            "decode_worker_id": worker_id,
                                        },
                                        "cached_tokens": 999,
                                    },
                                }
                            }
                        )
                        + "\n"
                    )

            self.assertEqual(
                latest_request_metrics(
                    str(prefix),
                    request_started_at_ms=1400,
                    request_completed_at_ms=1600,
                    timeout=0,
                ),
                (20, 20),
            )

    def test_latest_trace_reader_handles_missing_files_and_worker_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            self.assertEqual(
                latest_request_metrics(str(prefix), timeout=0),
                ("", ""),
            )

            with gzip.open(f"{prefix}.1.jsonl.gz", "wt", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "event": {
                                "event_type": "request_end",
                                "request": {"worker": {"prefill_worker_id": 0}},
                            }
                        }
                    )
                    + "\n"
                )

            self.assertEqual(
                latest_request_metrics(str(prefix)),
                (0, ""),
            )

    def test_latest_trace_reader_waits_for_a_new_request_end_event(self):
        old_event = ("previous-event", 10, 11)
        new_event = ("completed-request", 20, 21)
        with (
            patch("main._latest_request_end", side_effect=[old_event, new_event]),
            patch("main.time.sleep"),
        ):
            self.assertEqual(
                latest_request_metrics(
                    "./traces/rr", previous_event=old_event[0], timeout=1
                ),
                (20, 21),
            )

    def test_dashboard_uses_progress_total_for_planned_count(self):
        progress = Progress()
        progress.add_task("Requests processed", total=10)
        dashboard = build_dashboard(
            [],
            progress,
            "—",
            "READY",
            "",
            "",
            "round-robin",
        )
        output = io.StringIO()
        Console(file=output, width=140).print(dashboard)

        self.assertIn("R1 -> R10", output.getvalue())
        self.assertIn("Planned: 10", output.getvalue())

    def test_workload_has_thirty_sequential_identical_prompt_pairs(self):
        workload = build_requests()

        self.assertEqual(PROMPT_FAMILIES, 30)
        self.assertEqual(len(workload), 60)
        self.assertEqual([request[0] for request in workload], [
            f"R{index}" for index in range(1, 61)
        ])
        for family_index in range(PROMPT_FAMILIES):
            first, second = workload[family_index * 2 : family_index * 2 + 2]
            expected_family = f"F{family_index + 1:02d}"
            self.assertEqual(first[3:], (expected_family, "priming"))
            self.assertEqual(second[3:], (expected_family, "test"))
            self.assertEqual(first[2], second[2])
            self.assertEqual(first[2], f"FAMILY_{family_index:02d}\n\n{SHARED_PROMPT}")
        self.assertEqual(
            len({workload[index * 2][2] for index in range(PROMPT_FAMILIES)}),
            PROMPT_FAMILIES,
        )

    def test_chart_displays_latency_graph_metrics_without_explanation(self):
        output = io.StringIO()
        chart = LatencyChart(
            [
                {
                    "request_id": "R1",
                    "status": "success",
                    "ttft_ms": 10,
                    "total_ms": 20,
                    "cached_tokens": 17,
                    "prefix_hit_rate": 26,
                    "cache_on_worker": "w0",
                    "actual_prefill_worker": "w1",
                    "actual_decode_worker": "w2",
                },
                {
                    "request_id": "R2",
                    "status": "success",
                    "ttft_ms": 12,
                    "total_ms": 22,
                    "cached_tokens": "",
                    "prefix_hit_rate": "",
                    "cache_on_worker": "-",
                    "actual_prefill_worker": "",
                    "actual_decode_worker": "",
                }
            ]
        )

        Console(
            file=output,
            width=140,
            force_terminal=True,
            color_system="standard",
        ).print(chart)
        rendered = output.getvalue()

        self.assertIn("Request latency", rendered)
        self.assertIn("20.00 ms", rendered)
        self.assertIn("TTFT", rendered)
        self.assertIn("Total", rendered)
        self.assertIn("Cached", rendered)
        self.assertIn("tokens", rendered)
        self.assertIn("Prefix Hit", rendered)
        self.assertIn("Rate (%)", rendered)
        self.assertIn("(%)", rendered)
        self.assertIn("26.00", rendered)
        self.assertNotIn("KV hit rate", rendered)
        self.assertIn("Previous worker", rendered)
        self.assertIn("Router decision", rendered)
        self.assertIn("w2", rendered)
        self.assertNotIn("w1", rendered)
        self.assertNotIn("Common scale", rendered)
        self.assertNotIn("does not prove", rendered)

    def test_empty_chart_does_not_show_placeholder_text(self):
        output = io.StringIO()
        Console(file=output).print(LatencyChart([]))

        self.assertNotIn("Bars appear as requests complete.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
