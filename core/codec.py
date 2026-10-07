"""Lossy-but-bounded save encodings for face textures, and a columnar face-metadata format.

A ``TextureCodec`` quantizes a texture to integer multiples of ``step`` (optionally
in ``log(value + log_offset)`` space, optionally as a difference from another
texture), differences each texel row so smooth fields become small integers, and
narrows to the smallest integer dtype that fits.  zlib then compresses those far
better than raw float bits, whose low mantissa bits are noise.

Stages declare a codec on their ``TextureChannel`` (``codec=``), which registers it
here by name; ``core.persistence`` encodes every registered texture on save and
writes the codec parameters into the file, so loading needs no registry.
Unregistered textures are saved verbatim.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that degrades numpy scalars/arrays to plain Python."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


@dataclass(frozen=True)
class TextureCodec:
    """Save encoding for one texture; the reconstruction error is at most ``step / 2`` in coded space.

    ``log`` codes ``log(value + log_offset)`` — a relative step for fields spanning
    orders of magnitude (``step=0.002`` ≈ 0.1% maximum relative error).
    ``relative_to`` codes the difference from another (coded) texture, decoded first.
    """
    step: float
    log: bool = False
    log_offset: float = 0.0
    relative_to: Optional[str] = None


TEXTURE_CODECS: Dict[str, TextureCodec] = {}


def register_texture_codec(name: str, codec: TextureCodec) -> None:
    """Declare how texture ``name`` is saved; the same name must always get the same codec."""
    existing = TEXTURE_CODECS.get(name)
    if existing is not None and existing != codec:
        raise ValueError(f"conflicting codecs for texture {name!r}: {existing} vs {codec}")
    TEXTURE_CODECS[name] = codec


def _narrowest_int(values: np.ndarray) -> np.dtype:
    lo, hi = (int(values.min()), int(values.max())) if values.size else (0, 0)
    for dtype in (np.int8, np.int16, np.int32):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dtype)
    return np.dtype(np.int64)


def _forward(tex: np.ndarray, codec: TextureCodec, base: Optional[np.ndarray]) -> np.ndarray:
    values = tex.astype(np.float64)
    if codec.relative_to is not None:
        assert base is not None
        values = values - base.astype(np.float64)
    if codec.log:
        values = np.log(values + codec.log_offset)
    return values / codec.step


def encode_texture(tex: np.ndarray, codec: TextureCodec,
                   base: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Row-differenced integer codes for ``tex`` (axis 2 is the texel row), or ``None``
    if it can't be coded (non-finite values) and should be saved verbatim.

    ``base`` is the *decoded* ``relative_to`` texture, so decoding reproduces it exactly.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = _forward(tex, codec, base)
    if not np.all(np.isfinite(scaled)):
        return None
    codes = np.rint(scaled).astype(np.int64)
    diffs = np.diff(codes, axis=2, prepend=0)
    return diffs.astype(_narrowest_int(diffs))


def decode_texture(diffs: np.ndarray, codec: TextureCodec, dtype: np.dtype,
                   base: Optional[np.ndarray] = None) -> np.ndarray:
    """Invert ``encode_texture`` back to a texture of ``dtype``."""
    values = np.cumsum(diffs, axis=2, dtype=np.int64) * codec.step
    if codec.log:
        values = np.exp(values) - codec.log_offset
    if codec.relative_to is not None:
        assert base is not None
        values = values + base.astype(np.float64)
    return values.astype(dtype)


def codec_to_json(codec: TextureCodec, dtype: np.dtype) -> Dict[str, Any]:
    return {**dataclasses.asdict(codec), "dtype": np.dtype(dtype).str}


def codec_from_json(d: Dict[str, Any]) -> Tuple[TextureCodec, np.dtype]:
    fields = {f.name for f in dataclasses.fields(TextureCodec)}
    return TextureCodec(**{k: v for k, v in d.items() if k in fields}), np.dtype(d["dtype"])


def codec_order(names: List[str], codecs: Dict[str, TextureCodec]) -> List[str]:
    """``names`` with every ``relative_to`` base before the textures coded against it."""
    ordered: List[str] = []

    def visit(name: str, stack: Tuple[str, ...]) -> None:
        if name in ordered:
            return
        if name in stack:
            raise ValueError(f"cyclic texture codec bases: {stack + (name,)}")
        base = codecs[name].relative_to if name in codecs else None
        if base is not None and base in names:
            visit(base, stack + (name,))
        ordered.append(name)

    for n in names:
        visit(n, ())
    return ordered


# ── Columnar face metadata ───────────────────────────────────────────────────────


def json_bytes(obj: Any) -> np.ndarray:
    """``obj`` as a UTF-8 JSON ``uint8`` array (numpy ``str`` arrays are 4 bytes/char)."""
    return np.frombuffer(json.dumps(obj, cls=NumpyEncoder).encode("utf-8"), dtype=np.uint8)


def json_from_bytes(arr: np.ndarray) -> Any:
    return json.loads(arr.tobytes().decode("utf-8"))


def _column(values: List[Any]) -> Optional[Tuple[str, np.ndarray, Optional[List[str]]]]:
    """``(kind, array, vocab)`` for a uniform scalar column, else ``None``."""
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return "bool", np.array(values, dtype=np.bool_), None
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        ints = np.array(values, dtype=np.int64)
        return "int", ints.astype(_narrowest_int(ints)), None
    if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
           for v in values):
        return "float", np.array(values, dtype=np.float32), None
    if all(isinstance(v, str) for v in values):
        vocab = sorted(set(values))
        index = {s: i for i, s in enumerate(vocab)}
        codes = np.array([index[v] for v in values], dtype=np.int64)
        return "str", codes.astype(_narrowest_int(codes)), vocab
    return None


def encode_metadata(metadata: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> None:
    """Write ``metadata`` into ``arrays``: one typed ``meta_<key>`` column per key that every
    face holds as a uniform scalar (floats narrowed to float32), the rest as sparse JSON."""
    keys: List[str] = []
    for meta in metadata:
        for k in meta:
            if k not in keys:
                keys.append(k)
    header: Dict[str, Any] = {"n": len(metadata), "columns": {}, "sparse": {}}
    for key in keys:
        column = _column([m[key] for m in metadata]) if all(key in m for m in metadata) else None
        if column is None:
            header["sparse"][key] = {str(i): m[key] for i, m in enumerate(metadata) if key in m}
            continue
        kind, values, vocab = column
        arrays[f"meta_{key}"] = values
        header["columns"][key] = {"kind": kind, "vocab": vocab}
    arrays["face_metadata_header"] = json_bytes(header)


def decode_metadata(data: Any) -> List[Dict[str, Any]]:
    header = json_from_bytes(data["face_metadata_header"])
    metadata: List[Dict[str, Any]] = [{} for _ in range(header["n"])]
    for key, spec in header["columns"].items():
        values = data[f"meta_{key}"]
        if spec["kind"] == "str":
            vocab = spec["vocab"]
            column: List[Any] = [vocab[i] for i in values.tolist()]
        else:
            column = values.tolist()
        for meta, v in zip(metadata, column):
            meta[key] = v
    for key, entries in header["sparse"].items():
        for i, v in entries.items():
            metadata[int(i)][key] = v
    return metadata
