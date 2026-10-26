#!/usr/bin/env python3
"""TLS front for Flare, which serves plain HTTP only, with video as frames.

Terminates TLS on LISTEN (default 0.0.0.0:8888) with TLS_FRONT_CERT / TLS_FRONT_KEY and
a TLS 1.2 floor, and relays to Flare on UPSTREAM (default 127.0.0.1:8001). Responses
pass back byte for byte, so streaming is unchanged.

Requests are read one at a time. A POST to /v1/chat/completions whose body has a video
part goes through dsv41_video_frames.rewrite() (the module next to this file; the vLLM
recipe uses it as its middleware): each video_url becomes timestamped JPEG frames as image_url
parts, which Flare's DeepSeek frontend takes. Any other request is forwarded unchanged.
Chunked request bodies switch that connection to a plain relay. Standard library only,
plus OpenCV (python3-opencv) when a request has video.

Flare logs nothing per request, so the front does (stdout, the rank 0 journal): a `req`
line when a request goes upstream, with its shape and the requests in flight, and an
`end` line when its response is complete, with status, time to first byte, duration and
the usage counts. An error object in a response, also one inside a 200 stream, is
logged with its text. Health, metrics and model polls are not logged.
"""
import asyncio, collections, importlib.util, itertools, json, os, re, signal, ssl, time

LISTEN = os.environ.get("TLS_FRONT_LISTEN", "0.0.0.0:8888")
UPSTREAM = os.environ.get("TLS_FRONT_UPSTREAM", "127.0.0.1:8001")
CERT = os.environ.get("TLS_FRONT_CERT", "/etc/ssl/flare/fullchain.pem")
KEY = os.environ.get("TLS_FRONT_KEY", "/etc/ssl/flare/privkey.pem")
FRAMES = os.environ.get(
    "TLS_FRONT_VIDEO_MODULE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv41_video_frames.py"),
)
VIDEO_PATHS = {"/v1/chat/completions"}
QUIET_PATHS = {"/health", "/metrics", "/v1/models", "/version"}
_frames = None
_ids = itertools.count(1)
_inflight = 0
_USAGE = {key: re.compile(rb'"%s"\s*:\s*(\d+)' % key.encode())
          for key in ("prompt_tokens", "completion_tokens", "cached_tokens")}


def log(line):
    print(line, flush=True)


def describe(body):
    """Request shape for the log; never the prompt text."""
    try:
        request = json.loads(body)
    except ValueError:
        return "body=not-json"
    if not isinstance(request, dict):
        return "body=not-object"
    parts = []
    if "messages" in request:
        parts.append(f"msgs={len(request.get('messages') or ())}")
    if request.get("tools"):
        parts.append(f"tools={len(request['tools'])}")
    if "tool_choice" in request and not isinstance(request["tool_choice"], str):
        parts.append("tool_choice=forced")
    elif "tool_choice" in request:
        parts.append(f"tool_choice={request['tool_choice']}")
    parts.append(f"stream={int(bool(request.get('stream')))}")
    for key in ("max_tokens", "max_completion_tokens", "reasoning_effort", "n"):
        if key in request:
            parts.append(f"{key}={request[key]}")
    kwargs = request.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        parts.append(f"thinking={int(bool(kwargs['enable_thinking']))}")
    return " ".join(parts)


class Exchange:
    """One request on a connection, from forward to the end of its response."""

    def __init__(self, peer, method, path, size):
        self.id = next(_ids)
        self.peer, self.method, self.path, self.size = peer, method, path, size
        self.quiet = path in QUIET_PATHS
        self.start = time.monotonic()
        self.first = None  # first body byte: the first token of a stream
        self.sent = 0
        self.usage = {}
        self.error = None

    def opened(self, shape):
        global _inflight
        _inflight += 1
        if not self.quiet:
            log(f"req {self.id} {self.peer} {self.method} {self.path} {self.size}B {shape} "
                f"inflight={_inflight}")

    def scan(self, data):
        """Keep the last usage counts and the first error text that pass in the body."""
        for key, pattern in _USAGE.items():
            found = pattern.findall(data)
            if found:
                self.usage[key] = int(found[-1])
        if self.error is None and b'"error"' in data:
            at = data.find(b'"error"')
            self.error = data[at:at + 400].decode("utf-8", "replace").replace("\n", " ").strip()

    def closed(self, status):
        global _inflight
        _inflight -= 1
        if self.quiet and status == "200" and self.error is None:
            return
        now = time.monotonic()
        ttfb = f"{self.first - self.start:.2f}s" if self.first is not None else "-"
        usage = " ".join(f"{key.split('_')[0]}={value}" for key, value in self.usage.items())
        line = (f"end {self.id} {status} ttfb={ttfb} total={now - self.start:.2f}s "
                f"out={self.sent}B inflight={_inflight}")
        if usage:
            line += f" {usage}"
        if self.error:
            line += f" error={self.error}"
        log(line)


def frames():
    """Load the video module on first use (python3 -I keeps this directory off sys.path)."""
    global _frames
    if _frames is None:
        spec = importlib.util.spec_from_file_location("dsv41_video_frames", FRAMES)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _frames = module
    return _frames


def split(address):
    host, _, port = address.rpartition(":")
    return host, int(port)


def close(writer):
    try:
        writer.close()
    except Exception:
        pass


async def pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError):
        pass
    finally:
        close(writer)


def with_length(head, length):
    lines = [line for line in head.split(b"\r\n")
             if line and not line.lower().startswith(b"content-length:")]
    return b"\r\n".join(lines + [b"Content-Length: %d" % length]) + b"\r\n\r\n"


def error_response(message):
    body = json.dumps({"error": {"message": f"video input: {message}", "type": "BadRequestError",
                                 "param": None, "code": 400}}).encode()
    return (b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body)) + body


async def requests(client_reader, client_writer, upstream_writer, pending):
    """Forward each request, turning video parts into frames on the way."""
    peer = (client_writer.get_extra_info("peername") or ("?",))[0]
    try:
        while True:
            try:
                head = await client_reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                return
            lines = head.split(b"\r\n")
            method, _, rest = lines[0].partition(b" ")
            path = rest.partition(b" ")[0].split(b"?")[0].decode("latin-1")
            headers = {}
            for line in lines[1:]:
                name, _, value = line.partition(b":")
                headers[name.strip().lower()] = value.strip()
            if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
                exchange = Exchange(peer, method.decode("latin-1"), path, "chunked")
                pending.append(exchange)
                exchange.opened("body=chunked")
                upstream_writer.write(head)
                await pipe(client_reader, upstream_writer)
                return
            body = await client_reader.readexactly(int(headers.get(b"content-length", b"0") or 0))
            exchange = Exchange(peer, method.decode("latin-1"), path, len(body))
            if exchange.quiet or not body:
                shape = ""
            elif len(body) > 1 << 18:
                shape = await asyncio.to_thread(describe, body)
            else:
                shape = describe(body)
            if method == b"POST" and path in VIDEO_PATHS and b"video" in body:
                try:
                    module = frames()
                    new = await asyncio.to_thread(module.rewrite, body)
                except Exception as e:  # VideoError, bad JSON, or no OpenCV
                    log(f"req {exchange.id} {peer} {path} video rejected: {e or type(e).__name__}")
                    client_writer.write(error_response(str(e) or type(e).__name__))
                    await client_writer.drain()
                    return
                if new is not None:
                    body, head = new, with_length(head, len(new))
                    shape += f" video_rewrite={len(new)}B"
            # Queued before the write: the response can only follow it.
            pending.append(exchange)
            exchange.opened(shape)
            upstream_writer.write(head + body)
            await upstream_writer.drain()
    except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError, ValueError):
        pass
    finally:
        close(upstream_writer)


async def relay(reader, writer, exchange, size):
    """Copy size bytes of a response body, scanning them for usage and errors."""
    while size > 0:
        data = await reader.read(min(size, 1 << 16))
        if not data:
            raise asyncio.IncompleteReadError(b"", size)
        size -= len(data)
        if exchange.first is None:
            exchange.first = time.monotonic()
        exchange.sent += len(data)
        exchange.scan(data)
        writer.write(data)
        await writer.drain()


async def responses(upstream_reader, client_writer, pending):
    """Copy each response to the client, keeping its framing to close one exchange per response.

    Bodies pass on chunk by chunk as they arrive, so streaming is unchanged. A response
    without a length or chunked framing, or one that cannot be parsed, ends the framing:
    the rest of the connection is relayed as is.
    """
    exchange = None
    status = "---"
    try:
        while True:
            try:
                head = await upstream_reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                return
            client_writer.write(head)
            await client_writer.drain()
            lines = head.split(b"\r\n")
            status = lines[0].split(b" ")[1].decode("latin-1") if b" " in lines[0] else "???"
            if status.startswith("1"):
                continue  # 100 Continue; the final response follows
            exchange = pending.popleft() if pending else Exchange("?", "?", "?", 0)
            headers = {}
            for line in lines[1:]:
                name, _, value = line.partition(b":")
                headers[name.strip().lower()] = value.strip()
            if exchange.method == "HEAD" or status in ("204", "304"):
                pass
            elif b"chunked" in headers.get(b"transfer-encoding", b"").lower():
                while True:
                    line = await upstream_reader.readuntil(b"\r\n")
                    client_writer.write(line)
                    size = int(line.split(b";")[0].strip(), 16)
                    if size == 0:
                        while (trailer := await upstream_reader.readuntil(b"\r\n")) != b"\r\n":
                            client_writer.write(trailer)
                        client_writer.write(b"\r\n")
                        await client_writer.drain()
                        break
                    await relay(upstream_reader, client_writer, exchange, size)
                    tail = await upstream_reader.readexactly(2)
                    client_writer.write(tail)
            elif b"content-length" in headers:
                await relay(upstream_reader, client_writer, exchange,
                            int(headers[b"content-length"]))
            else:
                while data := await upstream_reader.read(1 << 16):
                    if exchange.first is None:
                        exchange.first = time.monotonic()
                    exchange.sent += len(data)
                    exchange.scan(data)
                    client_writer.write(data)
                    await client_writer.drain()
                exchange.closed(status)
                exchange = None
                return
            exchange.closed(status)
            exchange = None
    except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError, ValueError):
        pass
    finally:
        # A response cut off mid-body, and requests that got no response at all.
        if exchange is not None:
            exchange.closed(f"{status}-cut")
        while pending:
            pending.popleft().closed("no-response")
        close(client_writer)


_refused = 0


async def handle(client_reader, client_writer):
    global _refused
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(*split(UPSTREAM))
    except OSError:
        # Health polls hit this every few seconds while Flare loads: log the first only.
        if _refused == 0:
            log(f"upstream {UPSTREAM} refuses connections; answering 503 until it accepts")
        _refused += 1
        client_writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        close(client_writer)
        return
    if _refused:
        log(f"upstream {UPSTREAM} accepts connections again after {_refused} refused")
        _refused = 0
    pending = collections.deque()
    await asyncio.gather(requests(client_reader, client_writer, upstream_writer, pending),
                         responses(upstream_reader, client_writer, pending))


def context():
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(CERT, KEY)
    return tls


async def main():
    tls = context()
    server = await asyncio.start_server(handle, *split(LISTEN), ssl=tls, reuse_address=True,
                                        limit=1 << 20)
    loop = asyncio.get_running_loop()
    # A renewed certificate takes effect on SIGHUP without a restart.
    loop.add_signal_handler(signal.SIGHUP, lambda: tls.load_cert_chain(CERT, KEY))
    print(f"tls-front {LISTEN} -> {UPSTREAM}, video frames via {FRAMES}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
