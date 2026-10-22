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
"""
import asyncio, importlib.util, json, os, signal, ssl

LISTEN = os.environ.get("TLS_FRONT_LISTEN", "0.0.0.0:8888")
UPSTREAM = os.environ.get("TLS_FRONT_UPSTREAM", "127.0.0.1:8001")
CERT = os.environ.get("TLS_FRONT_CERT", "/etc/ssl/flare/fullchain.pem")
KEY = os.environ.get("TLS_FRONT_KEY", "/etc/ssl/flare/privkey.pem")
FRAMES = os.environ.get(
    "TLS_FRONT_VIDEO_MODULE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv41_video_frames.py"),
)
VIDEO_PATHS = {"/v1/chat/completions"}
_frames = None


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


async def requests(client_reader, client_writer, upstream_writer):
    """Forward each request, turning video parts into frames on the way."""
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
                upstream_writer.write(head)
                await pipe(client_reader, upstream_writer)
                return
            body = await client_reader.readexactly(int(headers.get(b"content-length", b"0") or 0))
            if method == b"POST" and path in VIDEO_PATHS and b"video" in body:
                try:
                    module = frames()
                    new = await asyncio.to_thread(module.rewrite, body)
                except Exception as e:  # VideoError, bad JSON, or no OpenCV
                    client_writer.write(error_response(str(e) or type(e).__name__))
                    await client_writer.drain()
                    return
                if new is not None:
                    body, head = new, with_length(head, len(new))
            upstream_writer.write(head + body)
            await upstream_writer.drain()
    except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError, ValueError):
        pass
    finally:
        close(upstream_writer)


async def handle(client_reader, client_writer):
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(*split(UPSTREAM))
    except OSError:
        client_writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        close(client_writer)
        return
    await asyncio.gather(requests(client_reader, client_writer, upstream_writer),
                         pipe(upstream_reader, client_writer))


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
