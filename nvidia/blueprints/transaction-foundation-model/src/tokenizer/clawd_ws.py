# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realtime Pump.fun tape client for https://clawd-ws.fly.dev/ ."""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import struct
import urllib.error
import urllib.request
from typing import Any, Iterable
from urllib.parse import urlparse

DEFAULT_HTTP = "https://clawd-ws.fly.dev"
DEFAULT_WS = "wss://clawd-ws.fly.dev/ws"
FRAME_TYPES = frozenset({"status", "token-launch", "token-enriched"})
USER_AGENT = "solana-clawd-trading-tokenizer/1.0"


def parse_pump_frame(raw: Any) -> dict[str, Any] | None:
    """Accept JSON type in {status, token-launch, token-enriched}."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        payload = raw
    else:
        return None
    frame_type = payload.get("type")
    if frame_type not in FRAME_TYPES:
        return None
    return payload


def fetch_health(url: str = DEFAULT_HTTP + "/health", timeout: float = 15.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError("health response was not a JSON object")
    return payload


def _mask(payload: bytes) -> bytes:
    key = os.urandom(4)
    return key + bytes(b ^ key[i % 4] for i, b in enumerate(payload))


def _read_exact(sock: ssl.SSLSocket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        piece = sock.recv(size - len(chunks))
        if not piece:
            raise ConnectionError("websocket closed")
        chunks.extend(piece)
    return bytes(chunks)


def _read_ws_frame(sock: ssl.SSLSocket) -> tuple[int, bytes]:
    header = _read_exact(sock, 2)
    opcode = header[0] & 0x0F
    masked = header[1] & 0x80
    length = header[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    mask_key = _read_exact(sock, 4) if masked else b""
    payload = _read_exact(sock, length)
    if masked:
        payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return opcode, payload


def recv_pump_frames(
    ws_url: str = DEFAULT_WS,
    *,
    timeout: float = 20.0,
    max_frames: int = 4,
    types: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Open the canonical tape (no query string, no subscribe frame)."""
    parsed = urlparse(ws_url)
    if parsed.scheme != "wss" or parsed.query:
        raise ValueError("canonical tape is wss://clawd-ws.fly.dev/ws with no query string")
    host = parsed.hostname or "clawd-ws.fly.dev"
    path = parsed.path or "/ws"
    wanted = frozenset(types or FRAME_TYPES)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    ctx = ssl.create_default_context()
    raw = socket.create_connection((host, parsed.port or 443), timeout=timeout)
    sock = ctx.wrap_socket(raw, server_hostname=host)
    sock.settimeout(timeout)
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"User-Agent: {USER_AGENT}\r\n"
        "\r\n"
    )
    try:
        sock.sendall(request.encode("ascii"))
        header_buf = b""
        while b"\r\n\r\n" not in header_buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            header_buf += chunk
        if b"\r\n\r\n" not in header_buf:
            raise RuntimeError("websocket handshake produced no HTTP header")
        header, leftover = header_buf.split(b"\r\n\r\n", 1)
        status_line = header.split(b"\r\n", 1)[0].decode("utf-8", "replace")
        if "101" not in status_line:
            raise RuntimeError(f"websocket handshake failed: {status_line}")
        leftover_view = leftover
        frames: list[dict[str, Any]] = []
        while leftover_view and len(frames) < max_frames:
            # leftover after HTTP is rare; fall through to framed reads
            leftover_view = b""
        while len(frames) < max_frames:
            opcode, payload = _read_ws_frame(sock)
            if opcode == 0x8:
                break
            if opcode == 0x9:
                sock.sendall(b"\x8a" + bytes([0x80 | len(payload)]) + _mask(payload))
                continue
            if opcode not in (0x1, 0x0):
                continue
            parsed_frame = parse_pump_frame(payload)
            if parsed_frame and parsed_frame.get("type") in wanted:
                frames.append(parsed_frame)
        return frames
    finally:
        try:
            sock.close()
        except OSError:
            pass


def frame_to_text(frame: dict[str, Any]) -> str:
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
