import subprocess
import time
import os
import signal
import sys
import requests

ITERATIONS = 10
HEALTH_URL = "http://localhost:8000/health"
HEALTH_TIMEOUT = 300
POLL_INTERVAL = 2


def wait_for_health(expected_count=4):
    print(f"Waiting for {expected_count} generate endpoints...")

    start = time.time()

    while time.time() - start < HEALTH_TIMEOUT:
        try:
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "etcd",
                    "etcdctl",
                    "get",
                    "",
                    "--from-key",
                    "--keys-only",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            endpoints = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith("v1/instances/dynamo/backend/generate/")
            ]

            count = len(endpoints)

            print(
                f"\rGenerate endpoints registered: {count}/{expected_count}",
                end="",
                flush=True,
            )

            if count >= expected_count:
                print("\nAll generate endpoints are active.")
                return

        except subprocess.SubprocessError as e:
            print(f"\nCould not query etcd: {e}")

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Only {count} generate endpoints registered after {HEALTH_TIMEOUT} seconds."
    )


def stop_dynamo(process):
    print("Stopping Dynamo...")

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)

    except subprocess.TimeoutExpired:
        print("Dynamo did not stop gracefully. Sending SIGKILL...")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()

    print("Dynamo stopped.")


def run_iteration(iteration):
    print(f"\n{'=' * 50}")
    print(f"ITERATION {iteration}/{ITERATIONS}")
    print(f"{'=' * 50}")

    dynamo = subprocess.Popen(["bash", "./agg_kv_router_4.sh"], start_new_session=True)

    try:
        wait_for_health()

        print("Running benchmark...")

        benchmark = subprocess.run(
            ["uv", "run", "main.py"], input="3\n", text=True, check=False
        )
        if benchmark.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed with exit code {benchmark.returncode}"
            )

        print(f"Iteration {iteration} completed successfully.")

    finally:
        stop_dynamo(dynamo)


def main():
    for i in range(1, ITERATIONS + 1):
        try:
            run_iteration(i)

        except Exception as e:
            print(f"Iteration {i} failed: {e}")
            sys.exit(1)

    print("\nAll 10 iterations completed successfully.")


if __name__ == "__main__":
    main()
