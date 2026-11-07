"""Tests for benchmark requests, trace metrics, and dashboard output."""

import io
import gzip
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from rich.progress import Progress

from config import REQUESTS
from dashboard import LatencyChart, build_dashboard
from logic import send_request
from main import (
    main as benchmark_main,
    _reset_prefix_caches,
    _run_benchmark_once,
    run_benchmark,
    _active_request_seconds,
    _label_worker_id,
    _request_prefix_hit_rate,
    _record_prompt_workers,
    _trace_prefix_for_experiment,
)
from prompts import (
    LONG_PROMPTS,
    PROMPT_FAMILIES,
    SHORT_PROMPTS,
    SHARED_PROMPT,
    build_requests,
)
from warmup import build_warmup_request, run_warmup, warm_worker
from worker_discovery import latest_request_trace, request_id_from_completion_id


class BenchmarkTests(unittest.TestCase):
    def test_warmup_request_uses_uuid_and_five_token_limit(self):
        warmup_id = "86a5a561-2f2c-4c8c-9059-6d8647da1bf6"
        request = build_warmup_request(8120, warmup_id)
        payload = json.loads(request.data)

        self.assertEqual(request.full_url, "http://localhost:8120/v1/chat/completions")
        self.assertEqual(payload["messages"][0]["content"], warmup_id)
        self.assertEqual(payload["max_tokens"], 5)
        self.assertIsNone(request.get_header("X-request-id"))

    def test_random_prompt_lists_have_ten_similarly_sized_prompts(self):
        for prompts in (SHORT_PROMPTS, LONG_PROMPTS):
            token_counts = [len(prompt.split()) for prompt in prompts]
            self.assertEqual(len(prompts), 10)
            self.assertTrue(all(count >= 64 for count in token_counts))
            self.assertLessEqual(max(token_counts) - min(token_counts), 3)
        self.assertLess(
            max(len(prompt.split()) for prompt in SHORT_PROMPTS),
            min(len(prompt.split()) for prompt in LONG_PROMPTS),
        )

    def test_random_prompting_selects_a_prompt_per_request_from_configured_length(self):
        with (
            patch("prompts.PROMPTING_TYPE", "random"),
            patch("prompts.PROMPT_LENGTH", "short"),
            patch("prompts.random.choice", side_effect=lambda prompts: prompts[0]) as choice,
        ):
            workload = build_requests(3)

        self.assertEqual(len(workload), REQUESTS)
        self.assertEqual(choice.call_count, REQUESTS)
        self.assertTrue(all(request[2] == SHORT_PROMPTS[0] for request in workload))
        self.assertTrue(all(request[1] == SHORT_PROMPTS[0] for request in workload))

    def test_priming_prompting_preserves_existing_workload(self):
        with patch("prompts.PROMPTING_TYPE", "priming"):
            workload = build_requests(1)

        self.assertEqual(workload[0][0], "RUN01-R1")
        self.assertEqual(workload[0][2], f"RUN_01_FAMILY_00\n\n{SHARED_PROMPT}")
        self.assertEqual(workload[0][2], workload[1][2])

    def test_trace_prefix_matches_selected_router(self):
        self.assertEqual(_trace_prefix_for_experiment("round-robin"), "./traces/rr")
        self.assertEqual(_trace_prefix_for_experiment("kv-router"), "./traces/kv")
        self.assertIsNone(_trace_prefix_for_experiment("cache-reuse"))

    def test_startup_warms_workers_before_interactive_menu_without_mapping(self):
        events = []
        with (
            patch("main.parse_args", return_value=None),
            patch(
                "main.run_warmup",
                side_effect=lambda: events.append(("warmup",)),
            ),
            patch(
                "main.choose_experiments",
                side_effect=lambda: events.append(("menu",)) or ["round-robin"],
            ),
            patch("main.run_benchmark", side_effect=lambda *args: events.append(("run",))),
            patch("main.console.print") as print_message,
        ):
            benchmark_main()

        self.assertEqual(events[0], ("warmup",))
        self.assertEqual(events[1], ("menu",))
        self.assertEqual(events[2][0], "run")
        self.assertTrue(
            any(
                "WARMING UP VLLM WORKERS" in str(call)
                for call in print_message.call_args_list
            )
        )

    def test_active_request_time_excludes_gaps_and_merges_overlap(self):
        results = [
            {"_request_started_at": 1.0, "_response_received_at": 2.0},
            {"_request_started_at": 4.0, "_response_received_at": 5.5},
            {"_request_started_at": 1.5, "_response_received_at": 3.0},
        ]

        self.assertEqual(_active_request_seconds(results), 3.5)

    def test_reset_prefix_cache_posts_to_every_worker_port(self):
        with (
            patch(
                "main.urllib.request.urlopen",
                side_effect=[
                    io.BytesIO(b'{"success": true}'),
                    io.BytesIO(b'{"success": true}'),
                ],
            ) as urlopen,
            patch("main.console.print"),
        ):
            _reset_prefix_caches({"w0": 8100, "w1": 8110})

        self.assertEqual(urlopen.call_count, 2)
        requests = [call.args[0] for call in urlopen.call_args_list]
        self.assertEqual(
            [(request.full_url, request.method) for request in requests],
            [
                ("http://localhost:8100/reset_prefix_cache", "POST"),
                ("http://localhost:8110/reset_prefix_cache", "POST"),
            ],
        )

    def test_reset_prefix_cache_rejects_unsuccessful_responses(self):
        with patch(
            "main.urllib.request.urlopen",
            return_value=io.BytesIO(b'{"success": false}'),
        ):
            with self.assertRaisesRegex(RuntimeError, "Cache reset failed"):
                _reset_prefix_caches({"w0": 8100})

    def test_benchmark_runs_ten_independent_workloads_and_resets_each_time(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "roundrobin.csv"
            with (
                patch("main.OUTPUT_CSV", str(output_path)),
                patch("main.PROMPTING_TYPE", "priming"),
                patch("main.CONCURRENCY", 1),
                patch("main.RUNS", 10),
                patch("main._reset_prefix_caches") as reset_caches,
                patch(
                    "main._run_benchmark_once",
                    side_effect=lambda experiment, labels, run: [
                        {
                            "run_number": run,
                            "request_id": f"RUN{run:02d}-R1",
                            "Mode": "priming",
                        }
                    ],
                ) as run_once,
                patch("main.console.print"),
            ):
                run_benchmark("round-robin", {})

            self.assertEqual(reset_caches.call_count, 10)
            self.assertEqual(
                [call.args[2] for call in run_once.call_args_list],
                list(range(1, 11)),
            )
            saved_path = output_path.with_name(f"priming_{output_path.name}")
            rows = saved_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 11)
            self.assertIn("RUN10-R1", rows[-1])
            self.assertEqual(rows[0], "run_number,request_id,Mode")

    def test_benchmark_saves_distinct_files_per_prompting_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "roundrobin.csv"
            for mode in ("priming", "random"):
                with (
                    patch("main.OUTPUT_CSV", str(output_path)),
                    patch("main.PROMPTING_TYPE", mode),
                    patch("main.CONCURRENCY", 1),
                    patch("main.RUNS", 1),
                    patch("main._reset_prefix_caches"),
                    patch(
                        "main._run_benchmark_once",
                        return_value=[
                            {"run_number": 1, "request_id": "RUN01-R1", "Mode": mode}
                        ],
                    ),
                    patch("main.console.print"),
                ):
                    run_benchmark("round-robin", {})

            priming_file = output_path.with_name("priming_roundrobin.csv")
            random_file = output_path.with_name("random_roundrobin.csv")
            self.assertTrue(priming_file.exists())
            self.assertTrue(random_file.exists())
            self.assertEqual(
                priming_file.read_text(encoding="utf-8").splitlines()[1].split(",")[-1],
                "priming",
            )
            self.assertEqual(
                random_file.read_text(encoding="utf-8").splitlines()[1].split(",")[-1],
                "random",
            )

    def test_benchmark_appends_rows_without_repeating_csv_header(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "roundrobin.csv"
            for run_number in (1, 2):
                with (
                    patch("main.OUTPUT_CSV", str(output_path)),
                    patch("main.PROMPTING_TYPE", "priming"),
                    patch("main.CONCURRENCY", 1),
                    patch("main.RUNS", 1),
                    patch("main._reset_prefix_caches"),
                    patch(
                        "main._run_benchmark_once",
                        return_value=[
                            {
                                "run_number": run_number,
                                "request_id": f"RUN{run_number:02d}-R1",
                                "Mode": "priming",
                            }
                        ],
                    ),
                    patch("main.console.print"),
                ):
                    run_benchmark("round-robin", {})

            saved_path = output_path.with_name("priming_roundrobin.csv")
            rows = saved_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows, [
                "run_number,request_id,Mode",
                "1,RUN01-R1,priming",
                "2,RUN02-R1,priming",
            ])

    def test_benchmark_uses_con_prefix_when_concurrency_exceeds_one(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "roundrobin.csv"
            with (
                patch("main.OUTPUT_CSV", str(output_path)),
                patch("main.PROMPTING_TYPE", "random"),
                patch("main.CONCURRENCY", 2),
                patch("main.RUNS", 1),
                patch("main._reset_prefix_caches"),
                patch(
                    "main._run_benchmark_once",
                    return_value=[
                        {"run_number": 1, "request_id": "RUN01-R1", "Mode": "random"}
                    ],
                ),
                patch("main.console.print"),
            ):
                run_benchmark("round-robin", {})

            self.assertTrue(
                output_path.with_name("con_random_roundrobin.csv").exists()
            )

    def test_concurrent_requests_are_numbered_by_family_dispatch_order(self):
        workload = [
            ("unused-R1", "question", "P01", "RUN01-F01", "priming"),
            ("unused-R2", "question", "T01", "RUN01-F01", "test"),
            ("unused-R3", "question", "P02", "RUN01-F02", "priming"),
            ("unused-R4", "question", "T02", "RUN01-F02", "test"),
        ]
        barrier = threading.Barrier(2)
        first_test_started = threading.Event()
        active = 0
        peak_active = 0
        active_lock = threading.Lock()
        submitted_requests = {}

        def fake_send_request(prompt, on_token=None, request_id=None):
            nonlocal active, peak_active
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
                submitted_requests[request_id] = prompt
            try:
                if prompt in {"P01", "P02"}:
                    barrier.wait(timeout=2)
                    if prompt == "P02":
                        self.assertTrue(first_test_started.wait(timeout=2))
                elif prompt == "T01":
                    first_test_started.set()
                now = time.perf_counter()
                return {
                    "trace_request_id": "test",
                    "ttft_ms": 1,
                    "response_ms": 1,
                    "total_ms": 2,
                    "request_started_at": now,
                    "response_received_at": now + 0.002,
                    "output": "",
                }
            finally:
                with active_lock:
                    active -= 1

        with (
            patch("main.build_requests", return_value=workload),
            patch("main.CONCURRENCY", 2),
            patch("main.send_request", side_effect=fake_send_request),
            patch(
                "main.latest_request_trace",
                return_value={
                    "ttft_ms": 1,
                    "total_time_ms": 2,
                    "input_tokens": 100,
                    "cached_tokens": 20,
                    "worker": {
                        "prefill_worker_id": 123,
                        "decode_worker_id": 123,
                    },
                },
            ) as trace_lookup,
            patch("main.Live"),
        ):
            rows = _run_benchmark_once(
                "round-robin",
                {"123": "w0"},
                1,
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual(peak_active, 2)
        self.assertEqual(
            submitted_requests,
            {
                "RUN01-R1": "P01",
                "RUN01-R2": "P02",
                "RUN01-R3": "T01",
                "RUN01-R4": "T02",
            },
        )
        self.assertTrue(all(row["prefix_hit_rate"] == "20.00" for row in rows))
        self.assertTrue(
            all(
                row["ttft_ms"] == 1
                and row["total_ms"] == 2
                and row["response_ms"] == 1
                and row["cached_tokens"] == 20
                and row["prompt_tokens"] == 100
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["router_decision_worker"] == row["actual_decode_worker"] == "w0"
                for row in rows
            )
        )
        self.assertEqual(
            {call.args[1] for call in trace_lookup.call_args_list},
            {"test"},
        )

    def test_worker_history_is_updated_only_after_request_completes(self):
        workload = [
            ("RUN01-R1", "question", "same prompt", "RUN01-F01", "priming"),
            ("RUN01-R2", "question", "same prompt", "RUN01-F01", "test"),
        ]
        request_processed = [False]
        record_calls = [0]
        record_prompt_workers = _record_prompt_workers

        def fake_send_request(prompt, on_token=None, request_id=None):
            request_processed[0] = True
            now = time.perf_counter()
            return {
                "ttft_ms": 1,
                "response_ms": 1,
                "total_ms": 2,
                "request_started_at": now,
                "response_received_at": now + 0.002,
                "output": "",
                "trace_request_id": "test",
            }

        def record_after_completion(*args):
            self.assertTrue(request_processed[0])
            if record_calls[0] == 0:
                self.assertNotIn(args[0], args[4])
            else:
                self.assertEqual(args[4][args[0]], ["w0"])
            record_calls[0] += 1
            return record_prompt_workers(*args)

        with (
            patch("main.build_requests", return_value=workload),
            patch("main.CONCURRENCY", 1),
            patch("main.send_request", side_effect=fake_send_request),
            patch(
                "main.latest_request_trace",
                return_value={
                    "ttft_ms": 1,
                    "total_time_ms": 2,
                    "input_tokens": 100,
                    "cached_tokens": 20,
                    "worker": {
                        "prefill_worker_id": 123,
                        "decode_worker_id": 123,
                    },
                },
            ),
            patch("main._record_prompt_workers", side_effect=record_after_completion),
            patch("main.Live"),
        ):
            rows = _run_benchmark_once("round-robin", {"123": "w0"}, 1)

        self.assertEqual(len(rows), 2)
        rows_by_request = {row["request_id"]: row for row in rows}
        self.assertEqual(rows_by_request["RUN01-R1"]["previous_workers"], "-")
        self.assertEqual(rows_by_request["RUN01-R2"]["previous_workers"], "w0")
        self.assertTrue(
            all(row["actual_prefill_worker"] == "w0" for row in rows)
        )

    def test_worker_ids_receive_stable_runtime_labels_in_first_seen_order(self):
        worker_labels = {}

        self.assertEqual(_label_worker_id(123456789, worker_labels), "w0")
        self.assertEqual(_label_worker_id(987654321, worker_labels), "w1")
        self.assertEqual(_label_worker_id(123456789, worker_labels), "w0")
        self.assertEqual(_label_worker_id(555555555, worker_labels), "w2")
        self.assertEqual(_label_worker_id("", worker_labels), "")
        self.assertEqual(
            worker_labels,
            {
                "123456789": "w0",
                "987654321": "w1",
                "555555555": "w2",
            },
        )

    def test_warmup_calls_direct_workers_without_reading_traces(self):
        events = []
        with (
            patch("warmup.WORKER_PORTS", {"w0": 8100, "w1": 8110}),
            patch("warmup.warm_worker", side_effect=lambda *args: events.append(args)),
            patch("builtins.print"),
        ):
            self.assertIsNone(run_warmup())
        self.assertEqual([(call[0], call[1]) for call in events], [
            ("w0", 8100),
            ("w1", 8110),
        ])

    def test_request_prefix_hit_rate_uses_request_cached_and_prompt_tokens(self):
        self.assertEqual(_request_prefix_hit_rate(192, 256), 75.0)
        self.assertIsNone(_request_prefix_hit_rate(192, 0))
        self.assertIsNone(_request_prefix_hit_rate("", 256))

    def test_request_reads_completion_id_from_stream(self):
        response = io.BytesIO(
            b'data: {"id":"cmpl-R7","choices":[{"delta":{"content":"ok"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        with patch("logic.urllib.request.urlopen", return_value=response) as urlopen:
            measurement = send_request("prompt", request_id="R7")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.get_header("X-request-id"), "R7")
        self.assertEqual(measurement["trace_request_id"], "R7")
        self.assertNotIn("stream_options", payload)

    def test_cache_column_reports_prior_prefill_and_decode_workers(self):
        worker_labels = {
            "123": "w0",
            "456": "w1",
            "789": "w2",
        }
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
                "prompt-a", 123, 456, worker_labels, prompt_workers
            ),
            ("w0", "w1", "w0/w2"),
        )
        self.assertEqual(
            _record_prompt_workers(
                "prompt-b", 456, 789, worker_labels, prompt_workers
            ),
            ("w1", "w2", "-"),
        )

    def test_completion_id_is_normalized_to_trace_request_id(self):
        self.assertEqual(
            request_id_from_completion_id("cmpl-request-1"),
            "request-1",
        )
        self.assertEqual(
            request_id_from_completion_id("chatcmpl-warmup-w1-request-1"),
            "warmup-w1-request-1",
        )
        self.assertEqual(
            request_id_from_completion_id(
                "cmpl-a89af172-2379-4bcb-ac48-eed92ac059f2"
            ),
            "a89af172-2379-4bcb-ac48-eed92ac059f2",
        )
        with self.assertRaises(ValueError):
            request_id_from_completion_id("request-1")

    def test_trace_reader_matches_completion_id_across_rotated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            for index, request_id, worker_id in (
                (1, "older-request", 10),
                (2, "request-1", 20),
            ):
                with gzip.open(
                    f"{prefix}.{index:06d}.jsonl.gz",
                    "wt",
                    encoding="utf-8",
                ) as trace:
                    trace.write(
                        json.dumps(
                            {
                                "timestamp": 123,
                                "event": {
                                    "schema": "dynamo.request.trace.v1",
                                    "event_type": "request_end",
                                    "request": {
                                        "request_id": request_id,
                                        "input_tokens": 25,
                                        "cached_tokens": 12,
                                        "ttft_ms": 3.5,
                                        "total_time_ms": 10.0,
                                        "worker": {
                                            "prefill_worker_id": worker_id,
                                            "decode_worker_id": worker_id,
                                        },
                                    },
                                },
                            }
                        )
                        + "\n"
                    )

            trace_request = latest_request_trace(
                str(prefix),
                "request-1",
                timeout=0,
            )
            chat_trace_request = latest_request_trace(
                str(prefix),
                "request-1",
                timeout=0,
            )

        self.assertEqual(trace_request["input_tokens"], 25)
        self.assertEqual(trace_request["cached_tokens"], 12)
        self.assertEqual(trace_request["worker"]["prefill_worker_id"], 20)
        self.assertEqual(chat_trace_request, trace_request)

    def test_trace_reader_does_not_fall_back_to_an_unrelated_request(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            with gzip.open(f"{prefix}.1.jsonl.gz", "wt", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "event": {
                                "event_type": "request_end",
                                "request": {
                                    "request_id": "unrelated",
                                    "worker": {
                                        "prefill_worker_id": 10,
                                        "decode_worker_id": 10,
                                    },
                                },
                            }
                        }
                    )
                    + "\n"
                )

            self.assertIsNone(
                latest_request_trace(str(prefix), "request-1", timeout=0)
            )

    def test_trace_reader_rejects_mismatched_prefill_and_decode_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rr"
            with gzip.open(f"{prefix}.1.jsonl.gz", "wt", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {
                            "event": {
                                "event_type": "request_end",
                                "request": {
                                    "request_id": "request-1",
                                    "worker": {
                                        "prefill_worker_id": 10,
                                        "decode_worker_id": 11,
                                    },
                                },
                            }
                        }
                    )
                    + "\n"
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "Prefill and decode worker IDs differ",
            ):
                latest_request_trace(str(prefix), "request-1", timeout=0)

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

        self.assertIn("RUNxx-R1 -> RUNxx-R10", output.getvalue())
        self.assertIn("Planned: 10", output.getvalue())

    def test_dashboard_shows_only_prompt_prefix_before_blank_line(self):
        progress = Progress()
        progress.add_task("Requests processed", total=1)
        dashboard = build_dashboard(
            [],
            progress,
            "R1",
            "RUNNING",
            "FAMILY_00\n\nShared benchmark prompt content",
            "Streaming output",
            "round-robin",
        )
        output = io.StringIO()
        Console(file=output, width=140).print(dashboard)

        self.assertIn("FAMILY_00", output.getvalue())
        self.assertNotIn("Shared benchmark prompt content", output.getvalue())

    def test_workload_has_thirty_sequential_identical_prompt_pairs_per_run(self):
        with patch("prompts.PROMPTING_TYPE", "priming"):
            workload = build_requests(1)
            second_run = build_requests(2)

        self.assertEqual(PROMPT_FAMILIES, REQUESTS // 2)
        self.assertEqual(len(workload), REQUESTS)
        self.assertEqual([request[0] for request in workload], [
            f"RUN01-R{index}" for index in range(1, REQUESTS + 1)
        ])
        for family_index in range(PROMPT_FAMILIES):
            first, second = workload[family_index * 2 : family_index * 2 + 2]
            expected_family = f"RUN01-F{family_index + 1:02d}"
            self.assertEqual(first[3:], (expected_family, "priming"))
            self.assertEqual(second[3:], (expected_family, "test"))
            self.assertEqual(first[2], second[2])
            self.assertEqual(
                first[2],
                f"RUN_01_FAMILY_{family_index:02d}\n\n{SHARED_PROMPT}",
            )
        self.assertEqual(
            len({workload[index * 2][2] for index in range(PROMPT_FAMILIES)}),
            PROMPT_FAMILIES,
        )
        self.assertTrue(all(request[0].startswith("RUN02-") for request in second_run))
        self.assertTrue(
            all(request[2] not in {item[2] for item in workload} for request in second_run)
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
                    "previous_workers": "w0",
                    "actual_prefill_worker": "w1",
                    "actual_decode_worker": "w2",
                    "router_decision_worker": "w2",
                },
                {
                    "request_id": "R2",
                    "status": "success",
                    "ttft_ms": 12,
                    "total_ms": 22,
                    "cached_tokens": "",
                    "prefix_hit_rate": "",
                    "previous_workers": "-",
                    "actual_prefill_worker": "",
                    "actual_decode_worker": "",
                    "router_decision_worker": "",
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
        self.assertIn("Previous workers", rendered)
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
