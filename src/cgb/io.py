"""Locate and open every CarbonGlobe array without copying it into RAM.

Two sources feed this project and they have the same shape grammar:

============  ==========================  =================================
source        array                       shape
============  ==========================  =================================
competition   ``test_x``                  (6171, 40, 12, 136)
competition   ``test_y0``                 (6171, 15, 7)
training      ``glob_X_fea.npy``          (54152, 40, 12, 136)
training      ``glob_Y{age}.npy``         (54152, 41, 12, 7)
============  ==========================  =================================

The training ``Y`` carries **41** year slots against the forcing's 40. Slot 0 is
the state before any forcing year is applied -- the training twin of ``test_y0``
-- and slots 1..40 are the state after each forcing year. That is what makes an
offline 40-step rollout possible at all, and it is measured in
``scripts/verify_train_shapes.py`` rather than assumed here.

Both competition ``.npz`` files store their members **uncompressed** (ZIP method
0), so :func:`open_test` memory-maps ``test_x`` straight out of the archive at
its member offset instead of materialising 1.6 GB. Verified by reading the
central directory: ``compress_type == 0`` and ``compress_size == file_size``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
TRAIN = DATA / "train"

#: Last axis of every state array, in order.
STATE_NAMES: tuple[str, ...] = ("height", "agb", "soil", "lai", "gpp", "npp", "rh")

#: Axis 1 of ``test_y0``, and the ``glob_Y{age}`` file suffixes.
SEED_AGES: tuple[int, ...] = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)

#: The two scored scenarios, in submission order.
SCENARIOS: tuple[str, ...] = ("ssp126", "ssp585")

#: Measured shapes. A file that disagrees is a changed release, not a file to
#: reshape into compliance.
TEST_X_SHAPE = (6171, 40, 12, 136)
TEST_Y0_SHAPE = (6171, 15, 7)
TRAIN_X_SHAPE = (54152, 40, 12, 136)
TRAIN_Y_SHAPE = (54152, 41, 12, 7)


def _mmap_zip_member(archive: Path, member: str) -> np.ndarray:
    """Memory-map one uncompressed ``.npy`` member of a ``.npz`` archive.

    The member body starts after its local file header, whose length depends on
    the stored filename and extra field -- both read here rather than assumed,
    because the central-directory extra field and the local one routinely differ.
    """
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{archive.name}:{member} is compressed; cannot memory-map it")
        header_offset = info.header_offset

    with open(archive, "rb") as raw:
        raw.seek(header_offset)
        local = raw.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(f"{archive.name}:{member} has no local file header at {header_offset}")
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
        body = header_offset + 30 + name_len + extra_len

        raw.seek(body)
        if raw.read(6) != b"\x93NUMPY":
            raise ValueError(f"{archive.name}:{member} does not start with the npy magic")
        major = raw.read(2)[0]
        size_width = 2 if major == 1 else 4
        header_len = int.from_bytes(raw.read(size_width), "little")
        descr = np.lib.format.read_array_header_1_0 if major == 1 else None
        del descr
        header = _parse_npy_header(raw.read(header_len))

    data_offset = body + 6 + 2 + size_width + header_len
    return np.memmap(
        archive,
        dtype=np.dtype(header["descr"]),
        mode="r",
        offset=data_offset,
        shape=tuple(header["shape"]),
    )


def _parse_npy_header(raw: bytes) -> dict[str, object]:
    """Parse an ``.npy`` header dict literal without ``eval``."""
    import ast

    header = ast.literal_eval(raw.decode("latin1").strip())
    if not isinstance(header, dict):
        raise ValueError("npy header is not a dict")
    if header.get("fortran_order"):
        raise ValueError("Fortran-ordered npy member cannot be memory-mapped as C order")
    return header


def open_test(scenario: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(test_x, test_y0)`` for one scenario, ``test_x`` memory-mapped."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")
    archive = DATA / f"test_{scenario}.npz"
    x = _mmap_zip_member(archive, "test_x.npy")
    with np.load(archive) as z:
        y0 = np.asarray(z["test_y0"])
    if x.shape != TEST_X_SHAPE:
        raise AssertionError(f"{archive.name}: test_x is {x.shape}, expected {TEST_X_SHAPE}")
    if y0.shape != TEST_Y0_SHAPE:
        raise AssertionError(f"{archive.name}: test_y0 is {y0.shape}, expected {TEST_Y0_SHAPE}")
    return x, y0


def open_train_x() -> np.ndarray:
    """Memory-map the present-day forcing, ``(54152, 40, 12, 136)``."""
    x = np.load(TRAIN / "glob_X_fea.npy", mmap_mode="r")
    if x.shape != TRAIN_X_SHAPE:
        raise AssertionError(f"glob_X_fea.npy is {x.shape}, expected {TRAIN_X_SHAPE}")
    return x


def open_train_y(age: int) -> np.ndarray:
    """Memory-map one seed age's present-day trajectory, ``(54152, 41, 12, 7)``."""
    if age not in SEED_AGES:
        raise ValueError(f"unknown seed age {age}, expected one of {SEED_AGES}")
    y = np.load(TRAIN / f"glob_Y{age:03d}.npy", mmap_mode="r")
    if y.shape != TRAIN_Y_SHAPE:
        raise AssertionError(f"glob_Y{age:03d}.npy is {y.shape}, expected {TRAIN_Y_SHAPE}")
    return y


def load_stats() -> dict[str, np.ndarray]:
    """Return the shipped driver and state normalisation statistics.

    ``y_mean[0]`` and ``y_mean[1]`` are the competition's submission divisors,
    which ``scripts/prove_metric.py`` checks rather than takes on trust.
    """
    with np.load(CACHE / "data_stats.npz") as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_split_indices() -> dict[str, np.ndarray]:
    """Return the paper's own ``train``/``val``/``test`` row indices into axis 0."""
    out: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        out[name] = np.loadtxt(CACHE / f"{name}.csv", dtype=np.int64, ndmin=1)
    return out
