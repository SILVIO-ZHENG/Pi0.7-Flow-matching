"""Asynchronous policy inference in a separate process."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from typing import Any

import numpy as np


def _policy_worker(
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    stop_event: mp.Event,
    remote_host: str,
    remote_port: int,
) -> None:
    """Connect to the policy server and run inference in a separate process."""
    try:
        from openpi_client import websocket_client_policy

        policy_client = websocket_client_policy.WebsocketClientPolicy(remote_host, remote_port)
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if request is None:
                break

            request_id = request["request_id"]
            observation = request["observation"]
            try:
                infer_start = time.monotonic()
                result = policy_client.infer(observation)
                client_infer_ms = (time.monotonic() - infer_start) * 1000
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": True,
                        "actions": np.asarray(result["actions"], dtype=np.float32),
                        "policy_timing": result.get("policy_timing", {}),
                        "model_timing": result.get("model_timing", {}),
                        "server_timing": result.get("server_timing", {}),
                        "client_timing": {
                            "websocket_infer_ms": client_infer_ms,
                        },
                    }
                )
            except Exception:
                response_queue.put(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": traceback.format_exc(),
                    }
                )
    except Exception:
        response_queue.put({"request_id": -1, "ok": False, "error": traceback.format_exc()})


class AsyncPolicyProcess:
    """Manage a policy inference process while retaining only the latest request."""

    def __init__(self, *, remote_host: str, remote_port: int) -> None:
        self._ctx = mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue(maxsize=1)
        self._response_queue: mp.Queue = self._ctx.Queue(maxsize=4)
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_policy_worker,
            args=(self._request_queue, self._response_queue, self._stop_event, remote_host, remote_port),
            daemon=True,
        )
        self._next_request_id = 0
        self._latest_request_id = -1
        self._inflight = False

    @property
    def inflight(self) -> bool:
        """Return whether an inference request is in flight."""
        return self._inflight

    def start(self) -> None:
        """Start the inference process."""
        self._process.start()

    def stop(self) -> None:
        """Stop the inference process."""
        self._stop_event.set()
        self._drop_stale_requests()
        with SuppressQueueFull():
            self._request_queue.put_nowait(None)
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

    def submit_latest(self, observation: dict[str, Any]) -> int:
        """Submit the latest observation and discard queued requests that have not started."""
        if not self._process.is_alive():
            raise RuntimeError("Policy inference subprocess is not running")
        self._drop_stale_requests()
        request_id = self._next_request_id
        self._next_request_id += 1
        self._latest_request_id = request_id
        self._request_queue.put({"request_id": request_id, "observation": observation})
        self._inflight = True
        return request_id

    def poll_latest(self) -> dict[str, Any] | None:
        """Return the newest inference result and discard older results."""
        latest = None
        while True:
            try:
                latest = self._response_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return None
        if int(latest.get("request_id", -1)) < 0 and not latest.get("ok", False):
            self._inflight = False
            return latest
        if latest.get("request_id", -1) >= self._latest_request_id:
            self._inflight = False
            return latest
        return None

    def _drop_stale_requests(self) -> None:
        while True:
            try:
                self._request_queue.get_nowait()
            except queue.Empty:
                break


class SuppressQueueFull:
    """Small context manager that ignores a full multiprocessing queue."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is queue.Full
