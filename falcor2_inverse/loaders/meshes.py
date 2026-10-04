"""Triangle mesh files: Wavefront OBJ, PLY (ASCII and binary) and Mitsuba's serialized format.

Each reader returns a dict with "positions" (V, 3), "faces" (F, 3), and optionally "texcoords" (V, 2) and "normals"
(V, 3). Polygons are fan-triangulated. Texture coordinates are flipped vertically (v -> 1 - v) for falcor2, whose
textures start at the top row.
"""

import struct
import zlib
from pathlib import Path

import numpy as np


def read_mesh(path, **options) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return read_obj(path, **options)
    if suffix == ".ply":
        return read_ply(path, **options)
    if suffix == ".serialized":
        return read_serialized(path, **options)
    raise ValueError(f"Unsupported mesh format {suffix!r}")


def read_obj(path, flip_tex_coords=True) -> dict:
    """OBJ positions, faces and (per-vertex, deduplicated by index triple) texture coordinates and normals."""
    v, vt, vn, corners, triangles = [], [], [], {}, []
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("vn "):
            vn.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            ids = []
            for token in line.split()[1:]:
                parts = (token.split("/") + ["", ""])[:3]
                key = tuple(int(p) if p else 0 for p in parts)
                key = tuple(i - 1 if i > 0 else n + i if i < 0 else -1 for i, n in zip(key, (len(v), len(vt), len(vn))))
                ids.append(corners.setdefault(key, len(corners)))
            triangles += [[ids[0], ids[k], ids[k + 1]] for k in range(1, len(ids) - 1)]
    keys = np.asarray(list(corners), np.int64).reshape(-1, 3)
    mesh = {"positions": np.asarray(v, np.float32)[keys[:, 0]], "faces": np.asarray(triangles, np.uint32)}
    if vt and (keys[:, 1] >= 0).all():
        uv = np.asarray(vt, np.float32)[keys[:, 1]]
        mesh["texcoords"] = np.stack([uv[:, 0], 1 - uv[:, 1]], 1) if flip_tex_coords else uv
    if vn and (keys[:, 2] >= 0).all():
        mesh["normals"] = np.asarray(vn, np.float32)[keys[:, 2]]
    return mesh


_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4", "float": "f4",
              "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4",
              "uint32": "u4", "float32": "f4", "float64": "f8"}


def read_ply(path, flip_tex_coords=True) -> dict:
    data = Path(path).read_bytes()
    end = data.index(b"end_header") + len(b"end_header")
    end = data.index(b"\n", end) + 1
    header = data[:end].decode("ascii", "ignore").splitlines()
    fmt = next(line.split()[1] for line in header if line.startswith("format"))
    elements = []
    for line in header:
        words = line.split()
        if words[:1] == ["element"]:
            elements.append((words[1], int(words[2]), []))
        elif words[:1] == ["property"]:
            elements[-1][2].append(words[1:])
    order = "<" if fmt == "binary_little_endian" else ">"
    body, offset, values = data[end:], 0, {}
    tokens = body.decode("ascii", "ignore").split() if fmt == "ascii" else None
    for name, count, props in elements:
        if all(p[0] != "list" for p in props):
            dtype = np.dtype([(p[1], order + _PLY_TYPES[p[0]]) for p in props])
            if tokens is not None:
                table = np.asarray(tokens[offset:offset + count * len(props)], np.float64).reshape(count, len(props))
                offset += count * len(props)
                values[name] = {p[1]: table[:, i] for i, p in enumerate(props)}
            else:
                array = np.frombuffer(body, dtype, count, offset)
                offset += dtype.itemsize * count
                values[name] = {p[1]: array[p[1]] for p in props}
            continue
        rows = []  # faces: a list property (possibly among others)
        for _ in range(count):
            row = None
            for p in props:
                if p[0] == "list":
                    if tokens is not None:
                        n = int(tokens[offset])
                        items = [int(x) for x in tokens[offset + 1:offset + 1 + n]]
                        offset += 1 + n
                    else:
                        count_type, item_type = np.dtype(order + _PLY_TYPES[p[1]]), np.dtype(order + _PLY_TYPES[p[2]])
                        n = int(np.frombuffer(body, count_type, 1, offset)[0])
                        offset += count_type.itemsize
                        items = np.frombuffer(body, item_type, n, offset).tolist()
                        offset += item_type.itemsize * n
                    if p[3] in ("vertex_indices", "vertex_index"):
                        row = items
                elif tokens is not None:
                    offset += 1
                else:
                    offset += np.dtype(_PLY_TYPES[p[0]]).itemsize
            rows.append(row)
        values[name] = {"faces": rows}
    vertex = values["vertex"]
    mesh = {"positions": np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float32),
            "faces": np.asarray([[f[0], f[k], f[k + 1]] for f in values.get("face", {}).get("faces", [])
                                 for k in range(1, len(f) - 1)], np.uint32).reshape(-1, 3)}
    for u, v in (("u", "v"), ("s", "t"), ("texture_u", "texture_v")):
        if u in vertex:
            mesh["texcoords"] = np.stack([vertex[u], 1 - vertex[v] if flip_tex_coords else vertex[v]], 1).astype(np.float32)
            break
    if "nx" in vertex:
        mesh["normals"] = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], 1).astype(np.float32)
    return mesh


def read_serialized(path, shape_index=0, flip_tex_coords=True) -> dict:
    """A mesh of a Mitsuba .serialized file (zlib-compressed shapes, version 3 or 4)."""
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, len(data) - 4)[0]
    version = struct.unpack_from("<H", data, 2)[0]
    if version >= 4:
        offsets = struct.unpack_from(f"<{count}Q", data, len(data) - 4 - 8 * count)
    else:
        offsets = struct.unpack_from(f"<{count}I", data, len(data) - 4 - 4 * count)
    stream = zlib.decompressobj().decompress(data[offsets[shape_index] + 4:])
    flags, offset = struct.unpack_from("<I", stream, 0)[0], 4
    if version >= 4:
        offset = stream.index(b"\0", offset) + 1  # the shape's name
    vertex_count, triangle_count = struct.unpack_from("<QQ", stream, offset)
    offset += 16
    real = np.float64 if flags & 0x2000 else np.float32

    def take(count, dtype):
        nonlocal offset
        array = np.frombuffer(stream, dtype, count, offset)
        offset += array.nbytes
        return array

    mesh = {"positions": take(3 * vertex_count, real).reshape(-1, 3).astype(np.float32)}
    if flags & 0x0001:
        mesh["normals"] = take(3 * vertex_count, real).reshape(-1, 3).astype(np.float32)
    if flags & 0x0002:
        uv = take(2 * vertex_count, real).reshape(-1, 2).astype(np.float32)
        mesh["texcoords"] = np.stack([uv[:, 0], 1 - uv[:, 1]], 1) if flip_tex_coords else uv
    if flags & 0x0008:
        take(3 * vertex_count, real)  # vertex colors
    index = np.uint64 if vertex_count > 0xFFFFFFFF else np.uint32
    mesh["faces"] = take(3 * triangle_count, index).reshape(-1, 3).astype(np.uint32)
    return mesh
