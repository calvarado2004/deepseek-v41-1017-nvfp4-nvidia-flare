#!/usr/bin/env python3
"""Report kernel arguments in a Flare/Hyperflare package whose pointer values fall inside the
rank's reserved address ranges but outside every recorded allocation.

The native runtime restores every allocation at its recorded address and replays kernel
arguments byte for byte, so such a pointer writes or reads unmapped (or later reused) memory.

Usage (on the host that holds the package, with the pin's venv Python, which has protobuf):
  $FLARE_SRC/.venv/bin/python check-unmapped.py PACKAGE_DIR [RANK] [--proto-dir DIR]
PROTO_DIR defaults to $FLARE_SRC/helios/proto (FLARE_SRC from the environment). Exit 1 on findings.
Known false positives: W4A8 MoE kernels pass constants such as 0x813000fa000; a value that also
appears in another package's graphs is not an address. Found 2026-11-11: the cuBLAS split-K
workspace of the first captured graph (patch 0012).
"""
import bisect
import collections
import json
import struct
import subprocess
import sys
from pathlib import Path

import os
import tempfile


def _load_protos():
    args = sys.argv[1:]
    proto_dir = None
    if "--proto-dir" in args:
        k = args.index("--proto-dir"); proto_dir = args[k + 1]; del args[k:k + 2]
    sys.argv[1:] = args
    proto_dir = Path(proto_dir or Path(os.environ.get("FLARE_SRC", ".")) / "helios" / "proto")
    out = tempfile.mkdtemp(prefix="flare-proto-")
    subprocess.run(["protoc", f"-I{proto_dir}", f"--python_out={out}", "runtime_core.proto", "core.proto"],
                   check=True, cwd=proto_dir)
    sys.path.insert(0, out)
    import core_pb2
    return core_pb2


pb = _load_protos()


def tensor(path, name):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    a, b = header[name]["data_offsets"]
    return raw[8 + n + a:8 + n + b]


def payload(path):
    data = tensor(path, "flare.payload")
    if data[:4] == bytes.fromhex("28b52ffd"):
        data = subprocess.run(["zstd", "-dq", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout
    return data


def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if c < 0x80:
            return r, i


def fields(b, i, end):
    while i < end:
        key, i = varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i); yield f, wt, v, None
        elif wt == 1:
            yield f, wt, int.from_bytes(b[i:i + 8], "little"), None; i += 8
        elif wt == 2:
            n, i = varint(b, i); yield f, wt, i, i + n; i += n
        elif wt == 5:
            yield f, wt, int.from_bytes(b[i:i + 4], "little"), None; i += 4
        else:
            raise ValueError(f"wire type {wt}")


def main():
    package = Path(sys.argv[1])
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    core = pb.ProtectedCore()
    core.ParseFromString(tensor(package / "model.safetensors", "flare.manifest"))
    body = pb.RankBody()
    body.ParseFromString(next(r.body for r in core.ranks if r.rank == rank))
    def extent(a):
        sized = a.sizing.unit_bytes * a.sizing.maximum if a.HasField("sizing") else 0
        return max(a.nbytes, sized, 1)
    allocs = sorted((a.address, a.address + extent(a), a.name) for a in body.allocations if a.HasField("address"))
    unaddressed = [a.name for a in body.allocations if not a.HasField("address")]
    if unaddressed:
        print("allocations without a recorded address:", unaddressed[:10], file=sys.stderr)
    starts = [a[0] for a in allocs]
    reservations = [(r.address, r.address + r.nbytes) for r in body.reservations]
    lo = min(r[0] for r in reservations)
    hi = max(r[1] for r in reservations)

    def owner(v):
        k = bisect.bisect_right(starts, v) - 1
        return allocs[k][2] if k >= 0 and v < allocs[k][1] else None

    def reserved(v):
        return any(a <= v < b for a, b in reservations)

    graphs_file = next(package.glob(f"deployments/*/rank-{rank}/graphs.safetensors"))
    b = payload(graphs_file)
    i = 0
    findings = collections.defaultdict(list)
    for g in body.graphs:
        for node in range(g.node_count + g.edge_count):
            n = int.from_bytes(b[i:i + 4], "little"); s, e = i + 4, i + 4 + n; i = e
            if node >= g.node_count:
                continue
            for f, wt, ks, ke in fields(b, s, e):
                if f != 1 or wt != 2:
                    continue  # not a kernel node
                func = None; args = []
                for kf, kwt, v, ve in fields(b, ks, ke):
                    if kf == 1: func = v
                    elif kf == 5: args.append((v, ve))
                for ai, (as_, ae) in enumerate(args):
                    for bf, bwt, ds, de in fields(b, as_, ae):
                        if bf != 1 or bwt != 2:
                            continue
                        for off in range(0, de - ds - 7, 8):
                            v = int.from_bytes(b[ds + off:ds + off + 8], "little")
                            if lo <= v < hi and reserved(v) and owner(v) is None:
                                findings[(g.name, func)].append((node, ai, off, v))
    if i != len(b):
        print(f"warning: parsed {i} of {len(b)} graph bytes", file=sys.stderr)
    print(f"rank {rank}: {len(body.graphs)} graphs, {len(allocs)} allocations, "
          f"{sum(len(v) for v in findings.values())} unmapped pointer arguments")
    for (graph, func), hits in sorted(findings.items()):
        fn = body.functions[func]
        values = sorted({hex(h[3]) for h in hits})
        print(f"  {graph:<28} {fn.symbol[:90]:<90} x{len(hits)} nodes {[h[0] for h in hits][:6]} values {values[:3]}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
