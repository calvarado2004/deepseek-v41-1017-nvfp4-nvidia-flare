#!/usr/bin/env python3
"""Make a Flare view of the DeepSeek-V4.1-Flash-1017 NVFP4 checkpoint, keeping its values.

Flare's DeepSeek V4.1 recipe and its native Helios row delivery read Engram tables
as F8_E4M3 rows of 256 bytes with a block-32 F8_E8M0 scale. The LibertAIDAI NVFP4
checkpoint stores them as packed E2M1 (U8 [rows, 128]) with the same block-32 E8M0
scale. Every E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) has an exact
E4M3 encoding, so this widening is exact: dequantized rows are bit-identical.

Flare rejects a tensor that appears in two shard files, so every source shard that
holds an Engram table is replaced as a whole. Output directory:
  - a symlink to every other source file, except model.safetensors.index.json
  - engram-L{layer}.safetensors: all tensors of that shard, with the table widened to
    F8_E4M3 and the other tensors copied byte for byte
  - model.safetensors.index.json pointing those tensors at the new files

Standard library only (the Sparks' system python has no numpy). Reads with pread and
drops every consumed and written range from the page cache: on GB10 the page cache
shares memory with the GPU of a live card. A widened table left by an earlier run of
this script (engram-fp8-L{layer}.safetensors) is copied instead of converted again.
"""
import argparse, hashlib, json, os, struct, sys, time

# E2M1 code -> E4M3 byte. E4M3: sign | 4-bit exponent (bias 7) | 3-bit mantissa.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_OF_MAGNITUDE = (0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C)
STEP = 1 << 27


def e4m3_value(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
    if exponent == 0:
        return sign * mantissa / 8 * 2.0 ** -6
    return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)


def code_to_e4m3(code):
    return (0x80 if code & 0x8 else 0) | E4M3_OF_MAGNITUDE[code & 0x7]


for _code in range(16):
    _expected = (-1.0 if _code & 0x8 else 1.0) * E2M1[_code & 0x7]
    assert e4m3_value(code_to_e4m3(_code)) == _expected, _code

# Packed byte holds element 2i in its low nibble and 2i+1 in its high nibble
# (confirmed against the original FP8 table: cosine 0.993 this way, ~0 swapped).
LOW = bytes(code_to_e4m3(b & 0xF) for b in range(256))
HIGH = bytes(code_to_e4m3(b >> 4) for b in range(256))


def header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(size))
    meta.pop("__metadata__", None)
    return 8 + size, meta


def drop(fd, offset, length, written=False):
    """Release a range from the page cache (Linux); a no-op where the calls are missing."""
    if written and hasattr(os, "fdatasync"):
        os.fdatasync(fd)
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)


def copy(src, src_offset, dst, dst_offset, length):
    done = 0
    while done < length:
        n = min(STEP, length - done)
        if hasattr(os, "copy_file_range"):
            moved = os.copy_file_range(src, dst, n, src_offset + done, dst_offset + done)
        else:
            data = os.pread(src, n, src_offset + done)
            moved = os.pwrite(dst, data, dst_offset + done)
        if moved <= 0:
            raise SystemExit(f"copy stopped at {src_offset + done}")
        drop(dst, dst_offset + done, moved, written=True)
        drop(src, src_offset + done, moved)
        done += moved


def widen(src, src_offset, dst, dst_offset, length, name):
    done, started = 0, time.time()
    while done < length:
        n = min(STEP, length - done)
        data = os.pread(src, n, src_offset + done)
        if len(data) != n:
            raise SystemExit(f"{name}: short read at {src_offset + done}")
        wide = bytearray(2 * n)
        wide[0::2] = data.translate(LOW)
        wide[1::2] = data.translate(HIGH)
        os.pwrite(dst, wide, dst_offset + 2 * done)
        drop(dst, dst_offset + 2 * done, 2 * n, written=True)
        drop(src, src_offset + done, n)
        done += n
        if done == length or (done // n) % 64 == 0:
            rate = done / max(time.time() - started, 1e-9) / 1e9
            print(f"{name}: {done / length:6.1%}  {rate:.2f} GB/s read", flush=True)


def rewrite_shard(src_path, dst_path, legacy, table):
    """Write all tensors of src_path to dst_path, widening `table` from U8 [rows, 128] to F8_E4M3."""
    base, meta = header(src_path)
    entry = meta[table]
    rows, packed = entry["shape"]
    if entry["dtype"] != "U8" or packed != 128:
        raise SystemExit(f"{table}: expected U8 [rows, 128], got {entry['dtype']} {entry['shape']}")
    order = sorted(meta, key=lambda k: meta[k]["data_offsets"][0])
    out, cursor = {}, 0
    for name in order:
        begin, end = meta[name]["data_offsets"]
        size = (end - begin) * (2 if name == table else 1)
        out[name] = {"dtype": "F8_E4M3" if name == table else meta[name]["dtype"],
                     "shape": [rows, 256] if name == table else meta[name]["shape"],
                     "data_offsets": [cursor, cursor + size]}
        cursor += size
    out["__metadata__"] = {"format": "pt", "source": os.path.basename(src_path) + " with " + table
                           + " widened exactly from E2M1 to E4M3"}
    text = json.dumps(out, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    out_base = 8 + len(text)
    reuse = None
    if os.path.exists(legacy):
        legacy_base, legacy_meta = header(legacy)
        if legacy_meta.get(table, {}).get("shape") == [rows, 256]:
            reuse = legacy_base + legacy_meta[table]["data_offsets"][0]
    tmp = dst_path + ".partial"
    src = os.open(src_path, os.O_RDONLY)
    dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    old = os.open(legacy, os.O_RDONLY) if reuse is not None else None
    try:
        os.write(dst, struct.pack("<Q", len(text)) + text)
        for name in order:
            begin, end = meta[name]["data_offsets"]
            target = out_base + out[name]["data_offsets"][0]
            if name != table:
                copy(src, base + begin, dst, target, end - begin)
            elif old is not None:
                print(f"{table}: copying the FP8 rows widened by an earlier run", flush=True)
                copy(old, reuse, dst, target, 2 * (end - begin))
            else:
                widen(src, base + begin, dst, target, end - begin, table)
        os.fsync(dst)
    finally:
        os.close(src)
        os.close(dst)
        if old is not None:
            os.close(old)
    os.rename(tmp, dst_path)
    return out


def verify(src_path, dst_path, table, samples=4096):
    """Compare sampled widened rows element by element, and every other tensor by SHA-256."""
    base, meta = header(src_path)
    out_base, out = header(dst_path)
    rows = meta[table]["shape"][0]
    with open(src_path, "rb") as s, open(dst_path, "rb") as d:
        for i in range(samples):
            row = (i * 2654435761) % rows if i else rows - 1
            s.seek(base + meta[table]["data_offsets"][0] + row * 128)
            packed = s.read(128)
            d.seek(out_base + out[table]["data_offsets"][0] + row * 256)
            wide = d.read(256)
            for j, byte in enumerate(packed):
                for k, code in enumerate((byte & 0xF, byte >> 4)):
                    expected = (-1.0 if code & 0x8 else 1.0) * E2M1[code & 0x7]
                    if e4m3_value(wide[2 * j + k]) != expected:
                        raise SystemExit(f"{table}: row {row} element {2 * j + k} differs")
        for name in meta:
            if name == table:
                continue
            digests = []
            for f, b, offsets in ((s, base, meta[name]["data_offsets"]), (d, out_base, out[name]["data_offsets"])):
                h, left = hashlib.sha256(), offsets[1] - offsets[0]
                f.seek(b + offsets[0])
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    h.update(chunk)
                    left -= len(chunk)
                digests.append(h.digest())
            if digests[0] != digests[1]:
                raise SystemExit(f"{name}: copied bytes differ")
    print(f"{os.path.basename(dst_path)}: {samples} sampled rows of {table} identical, "
          f"{len(meta) - 1} other tensors byte-identical", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source, output = os.path.abspath(args.source), os.path.abspath(args.output)
    index = json.load(open(os.path.join(source, "model.safetensors.index.json")))
    config = json.load(open(os.path.join(source, "config.json")))
    if config.get("quantization_config", {}).get("engram_embed_quant") != "mxfp4":
        raise SystemExit("source Engram tables are not MXFP4; Flare reads them as they are")
    weight_map = index["weight_map"]
    tables = {name: weight_map[name] for name in weight_map if name.endswith(".engram.embed.weight")}
    replaced = set(tables.values())
    if len(replaced) != len(tables):
        raise SystemExit("two Engram tables share one shard")
    os.makedirs(output, exist_ok=True)
    for entry in sorted(os.listdir(source)):
        link = os.path.join(output, entry)
        if entry in replaced:
            if os.path.islink(link):
                os.unlink(link)
        elif entry != "model.safetensors.index.json" and not os.path.lexists(link):
            os.symlink(os.path.join(source, entry), link)
    total = index["metadata"]["total_size"]
    for table, shard in sorted(tables.items()):
        layer = table.split(".")[1]
        src_path = os.path.join(source, shard)
        file = f"engram-L{layer}.safetensors"
        dst_path = os.path.join(output, file)
        legacy = os.path.join(output, f"engram-fp8-L{layer}.safetensors")
        if os.path.exists(dst_path):
            print(f"{file} exists, verifying only", flush=True)
            out = header(dst_path)[1]
        else:
            out = rewrite_shard(src_path, dst_path, legacy, table)
        verify(src_path, dst_path, table)
        for name in out:
            if name != "__metadata__":
                weight_map[name] = file
        total += header(src_path)[1][table]["shape"][0] * 128
        if os.path.exists(legacy):
            os.unlink(legacy)
    index["metadata"]["total_size"] = total
    with open(os.path.join(output, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"done: {output} total_size {total}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
