"""Video input for DeepSeek-V4.1 as ordered frames (vLLM --middleware).

DeepSeek-V4.1 has a vision tower but no video token: vLLM's V4.1 encoder
rejects any content part that is not text or an image. This ASGI middleware
rewrites POST /v1/chat/completions bodies before vLLM parses them. Each
``video_url`` part becomes a short header and N timestamped JPEG frames as
``image_url`` parts, sampled evenly over the clip, so the model can reason
across time. Requests without video pass through byte for byte.

    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,...",
                                        "num_frames": 12, "fps": 1.0}}

``num_frames`` / ``fps`` are optional per part. Without them the clip gets
DSV41_VIDEO_FPS frames per second, between 2 and DSV41_VIDEO_MAX_FRAMES. All
videos and images of one request together stay within
DSV41_VIDEO_MAX_IMAGES (set it to the image cap of --limit-mm-per-prompt);
frames are cut evenly to fit, and a request that cannot fit gets a 400.

Load with: --middleware dsv41_video_frames.VideoFramesMiddleware
(the launcher mounts this file and puts its directory on PYTHONPATH).
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import os
import socket
import tempfile
import urllib.parse
import urllib.request

PATHS = {"/v1/chat/completions"}
FPS = float(os.environ.get("DSV41_VIDEO_FPS", "1.0"))
MAX_FRAMES = int(os.environ.get("DSV41_VIDEO_MAX_FRAMES", "16"))
MAX_IMAGES = int(os.environ.get("DSV41_VIDEO_MAX_IMAGES", "32"))
MAX_SIDE = int(os.environ.get("DSV41_VIDEO_MAX_SIDE", "1024"))
MAX_BYTES = int(os.environ.get("DSV41_VIDEO_MAX_BYTES", str(512 << 20)))
JPEG_QUALITY = int(os.environ.get("DSV41_VIDEO_JPEG_QUALITY", "90"))
# Remote URLs are off by default: the server would fetch whatever a client
# names. Set 1 to allow http(s) URLs that resolve to public addresses.
ALLOW_REMOTE = os.environ.get("DSV41_VIDEO_ALLOW_REMOTE", "0") == "1"
TIMEOUT_S = float(os.environ.get("DSV41_VIDEO_FETCH_TIMEOUT_S", "60"))


class VideoError(ValueError):
    pass


def _fetch(url: str) -> bytes:
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        if ";base64" not in head:
            raise VideoError("video data URL must be base64")
        raw = base64.b64decode(data, validate=False)
    elif url.startswith(("http://", "https://")):
        if not ALLOW_REMOTE:
            raise VideoError("remote video URLs are disabled; send a data: URL")
        host = urllib.parse.urlsplit(url).hostname or ""
        for info in socket.getaddrinfo(host, None):
            if not ipaddress.ip_address(info[4][0]).is_global:
                raise VideoError("video URL resolves to a non-public address")
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            raw = r.read(MAX_BYTES + 1)
    else:
        raise VideoError("video_url must be a data: or http(s) URL")
    if len(raw) > MAX_BYTES:
        raise VideoError(f"video larger than {MAX_BYTES} bytes")
    return raw


def _frames(raw: bytes, num_frames: int | None, fps: float | None, cap_n: int):
    """[(t_seconds, jpeg_bytes)], evenly spaced, plus the clip duration."""
    import cv2  # in the vLLM image (video IO dependency)

    with tempfile.NamedTemporaryFile(suffix=".video") as f:
        f.write(raw)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        if not cap.isOpened():
            raise VideoError("could not decode the video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        rate = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total <= 0 or rate <= 0:
            # Container without an index: decode everything once to count.
            total = 0
            while cap.grab():
                total += 1
            cap.release()
            cap = cv2.VideoCapture(f.name)
            rate = rate if rate > 0 else 25.0
        if total <= 0:
            raise VideoError("video has no frames")
        duration = total / rate
        n = num_frames or math.ceil(duration * (fps or FPS))
        n = max(1, min(n, MAX_FRAMES, cap_n, total))
        if n < 2 and min(MAX_FRAMES, cap_n, total) >= 2:
            n = 2
        # Centre of n equal slices, so the first and last frames are not
        # black fade-in/out frames.
        picks = [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]
        out = []
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                out.append((idx / rate, jpg.tobytes()))
        cap.release()
    if not out:
        raise VideoError("could not read any frame of the video")
    return out, duration


def _video_part(part: dict) -> dict | None:
    t = part.get("type")
    if t == "video_url":
        v = part.get("video_url")
        return v if isinstance(v, dict) else {"url": v}
    if t == "video" and isinstance(part.get("video"), (str, dict)):
        v = part["video"]
        return v if isinstance(v, dict) else {"url": v}
    return None


def rewrite(body: bytes) -> bytes | None:
    """The rewritten body, or None when the request has no video part."""
    if b"video" not in body:
        return None
    req = json.loads(body)
    msgs = req.get("messages")
    if not isinstance(msgs, list):
        return None
    videos, images = [], 0
    for m in msgs:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for p in content:
            if not isinstance(p, dict):
                continue
            if _video_part(p) is not None:
                videos.append(p)
            elif p.get("type") in ("image_url", "image", "input_image"):
                images += 1
    if not videos:
        return None
    budget = MAX_IMAGES - images
    if budget < len(videos):
        raise VideoError(
            f"{len(videos)} video(s) and {images} image(s) exceed the {MAX_IMAGES}-image cap"
        )
    share = budget // len(videos)
    for m in msgs:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        new = []
        for p in content:
            v = _video_part(p) if isinstance(p, dict) else None
            if v is None:
                new.append(p)
                continue
            frames, duration = _frames(
                _fetch(str(v.get("url", ""))),
                int(v["num_frames"]) if v.get("num_frames") else None,
                float(v["fps"]) if v.get("fps") else None,
                share,
            )
            times = ", ".join(f"{t:.1f}" for t, _ in frames)
            new.append({"type": "text", "text": (
                f"[Video: {duration:.1f} s clip shown as {len(frames)} frames in time "
                f"order, sampled at t = {times} s.]")})
            for i, (t, jpg) in enumerate(frames, 1):
                new.append({"type": "text", "text": f"Frame {i} (t = {t:.1f} s):"})
                new.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
            new.append({"type": "text", "text": "[End of video.]"})
        m["content"] = new
    return json.dumps(req).encode()


class VideoFramesMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") not in PATHS:
            await self.app(scope, receive, send)
            return
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        try:
            new = await asyncio.to_thread(rewrite, body)
        except (VideoError, ValueError, KeyError, TypeError) as e:
            err = json.dumps({"error": {"message": f"video input: {e}", "type": "BadRequestError",
                                        "param": None, "code": 400}}).encode()
            await send({"type": "http.response.start", "status": 400,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(err)).encode())]})
            await send({"type": "http.response.body", "body": err})
            return
        if new is not None:
            body = new
            scope = dict(scope)
            scope["headers"] = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
            scope["headers"].append((b"content-length", str(len(body)).encode()))
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
