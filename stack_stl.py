#!/usr/bin/env python3
"""Stack one or more STL files vertically to create printable copies.

Example:
    python stack_stl.py input1.stl input2.stl output.stl --clones 3 --gap 0.3 --center

Use ``--center`` to align each input around the XY axis before stacking.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

Vector3 = Tuple[float, float, float]
Triangle = Tuple[Vector3, Tuple[Vector3, Vector3, Vector3]]


def compute_bounds(
    triangles: Sequence[Triangle],
) -> Tuple[float, float, float, float, float, float]:
    """Return min/max bounds for X, Y, and Z coordinates."""

    vertex_iter = (
        vertex for _, vertices in triangles for vertex in vertices
    )

    try:
        first_vertex = next(vertex_iter)
    except StopIteration as exc:
        raise ValueError("no vertices available to compute bounds") from exc

    min_x = max_x = first_vertex[0]
    min_y = max_y = first_vertex[1]
    min_z = max_z = first_vertex[2]

    for vx, vy, vz in vertex_iter:
        if vx < min_x:
            min_x = vx
        if vx > max_x:
            max_x = vx
        if vy < min_y:
            min_y = vy
        if vy > max_y:
            max_y = vy
        if vz < min_z:
            min_z = vz
        if vz > max_z:
            max_z = vz

    return min_x, max_x, min_y, max_y, min_z, max_z


def translate_triangles(
    triangles: Sequence[Triangle],
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
) -> List[Triangle]:
    """Translate all triangle vertices by the given offsets."""

    if not triangles:
        return []

    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return [(normal, vertices) for normal, vertices in triangles]

    translated: List[Triangle] = []
    for normal, vertices in triangles:
        translated_vertices = tuple(
            (vx + dx, vy + dy, vz + dz) for (vx, vy, vz) in vertices
        )
        translated.append((normal, translated_vertices))

    return translated


def normalize_triangles(
    triangles: Sequence[Triangle],
    *,
    center_xy: bool = False,
) -> List[Triangle]:
    """Align triangles so the base sits at Z=0 and optionally center in XY."""

    if not triangles:
        return []

    min_x, max_x, min_y, max_y, min_z, _ = compute_bounds(triangles)
    dx = -((min_x + max_x) * 0.5) if center_xy else 0.0
    dy = -((min_y + max_y) * 0.5) if center_xy else 0.0
    dz = -min_z

    return translate_triangles(triangles, dx=dx, dy=dy, dz=dz)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer value for CLI arguments."""

    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def non_negative_float(value: str) -> float:
    """Parse a non-negative floating point value for CLI arguments."""

    number = float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def load_stl(path: Path) -> Tuple[List[Triangle], dict]:
    """Load an STL file and return its triangles and metadata."""

    data = path.read_bytes()
    binary_result = _try_read_binary_stl(data)
    if binary_result is not None:
        header, triangles = binary_result
        return triangles, {"format": "binary", "header": header}

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "STL does not appear to be valid binary or UTF-8 encoded ASCII"
        ) from exc

    name, triangles = _read_ascii_stl(text)
    return triangles, {"format": "ascii", "name": name}


def _try_read_binary_stl(data: bytes) -> Tuple[bytes, List[Triangle]] | None:
    """Attempt to interpret the given data as a binary STL file."""

    if len(data) < 84:
        return None

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size != len(data):
        return None

    triangles: List[Triangle] = []
    offset = 84
    for _ in range(triangle_count):
        chunk = data[offset : offset + 50]
        if len(chunk) != 50:
            return None

        normal = struct.unpack_from("<3f", chunk, 0)
        v1 = struct.unpack_from("<3f", chunk, 12)
        v2 = struct.unpack_from("<3f", chunk, 24)
        v3 = struct.unpack_from("<3f", chunk, 36)
        triangles.append(
            (
                (float(normal[0]), float(normal[1]), float(normal[2])),
                (
                    (float(v1[0]), float(v1[1]), float(v1[2])),
                    (float(v2[0]), float(v2[1]), float(v2[2])),
                    (float(v3[0]), float(v3[1]), float(v3[2])),
                ),
            )
        )
        offset += 50

    return data[:80], triangles


def _read_ascii_stl(text: str) -> Tuple[str, List[Triangle]]:
    """Parse an ASCII STL file from text."""

    lines = text.splitlines()
    name = "stacked"
    if lines:
        first = lines[0].strip()
        if first.lower().startswith("solid"):
            name = first[5:].strip() or "stacked"

    triangles: List[Triangle] = []
    current_normal: Vector3 = (0.0, 0.0, 0.0)
    current_vertices: List[Vector3] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("facet normal"):
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid facet normal line: {line}")
            current_normal = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        elif lower.startswith("vertex"):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid vertex line: {line}")
            vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
            current_vertices.append(vertex)
        elif lower.startswith("endfacet"):
            if len(current_vertices) != 3:
                raise ValueError("Facet does not contain exactly 3 vertices")
            triangles.append(
                (
                    current_normal,
                    (
                        current_vertices[0],
                        current_vertices[1],
                        current_vertices[2],
                    ),
                )
            )
            current_vertices = []

    if current_vertices:
        raise ValueError("Dangling vertex data detected in ASCII STL")
    if not triangles:
        raise ValueError("No triangles found in STL file")

    return name, triangles


def stack_triangles(
    triangles: Sequence[Triangle],
    clones: int,
    gap: float,
) -> List[Triangle]:
    """Create stacked copies of the given triangles along the Z axis."""

    try:
        _, _, _, _, min_z, max_z = compute_bounds(triangles)
    except ValueError as exc:
        raise ValueError("Input STL has no vertices to stack") from exc

    height = max_z - min_z
    normalized = translate_triangles(triangles, dz=-min_z)

    stacked: List[Triangle] = []
    current_top: float | None = None

    for _ in range(clones):
        offset = 0.0 if current_top is None else current_top + gap
        stacked.extend(translate_triangles(normalized, dz=offset))
        current_top = offset + height

    return stacked


def stack_models_vertically(
    triangle_groups: Sequence[Sequence[Triangle]],
    gap: float,
) -> List[Triangle]:
    """Stack multiple STL models vertically with an optional gap between each."""

    stacked: List[Triangle] = []
    current_top: float | None = None

    for triangles in triangle_groups:
        if not triangles:
            continue

        try:
            _, _, _, _, min_z, max_z = compute_bounds(triangles)
        except ValueError:
            continue

        if current_top is None:
            offset = -min_z
        else:
            offset = current_top + gap - min_z
        stacked.extend(translate_triangles(triangles, dz=offset))
        current_top = offset + max_z

    return stacked


def write_binary_stl(path: Path, header: bytes, triangles: Sequence[Triangle]) -> None:
    """Write triangles to a binary STL file."""

    prepared_header = (header or b"").ljust(80, b"\0")[:80]
    with path.open("wb") as handle:
        handle.write(prepared_header)
        handle.write(struct.pack("<I", len(triangles)))
        for normal, vertices in triangles:
            handle.write(
                struct.pack(
                    "<12fH",
                    normal[0],
                    normal[1],
                    normal[2],
                    vertices[0][0],
                    vertices[0][1],
                    vertices[0][2],
                    vertices[1][0],
                    vertices[1][1],
                    vertices[1][2],
                    vertices[2][0],
                    vertices[2][1],
                    vertices[2][2],
                    0,
                )
            )


def write_ascii_stl(path: Path, name: str, triangles: Sequence[Triangle]) -> None:
    """Write triangles to an ASCII STL file."""

    mesh_name = name.strip() if name else "stacked"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"solid {mesh_name}\n")
        for normal, vertices in triangles:
            handle.write(
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
            )
            handle.write("    outer loop\n")
            for vertex in vertices:
                handle.write(
                    f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n"
                )
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write(f"endsolid {mesh_name}\n")


def ensure_output_directory(path: Path) -> None:
    """Create the parent directories for the output file if needed."""

    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the stacking script."""

    parser = argparse.ArgumentParser(
        description=(
            "Stack one or more STL files vertically, then repeat the stack with a gap."
        )
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="one or more input STL files followed by the output STL path",
    )
    parser.add_argument(
        "--clones",
        "-c",
        type=positive_int,
        default=1,
        help="total number of vertically stacked copies to produce",
    )
    parser.add_argument(
        "--gap",
        "-g",
        type=non_negative_float,
        default=0.3,
        help="vertical gap between each copy (same units as the STL)",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="center each input model around the XY origin before stacking",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "ascii", "binary"],
        default="auto",
        help="encoding for the output STL (default: match input)",
    )
    args = parser.parse_args(argv)

    if len(args.paths) < 2:
        parser.error("at least one input STL and an output path are required")

    args.inputs = args.paths[:-1]
    args.output = args.paths[-1]

    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    triangle_groups: List[List[Triangle]] = []
    metadata_list: List[dict] = []

    for input_path in args.inputs:
        triangles, metadata = load_stl(input_path)
        normalized = normalize_triangles(triangles, center_xy=args.center)
        triangle_groups.append(normalized)
        metadata_list.append(metadata)

    base_stack = stack_models_vertically(triangle_groups, args.gap)
    stacked = stack_triangles(base_stack, args.clones, args.gap)

    primary_metadata = metadata_list[0]

    output_format = args.output_format
    if output_format == "auto":
        output_format = primary_metadata.get("format", "ascii")

    ensure_output_directory(args.output)

    if output_format == "binary":
        header = primary_metadata.get("header", b"Stacked by stack_stl.py")
        write_binary_stl(args.output, header, stacked)
    else:
        name = primary_metadata.get("name", args.inputs[0].stem)
        if len(args.inputs) > 1:
            name = f"{name}_stacked"
        write_ascii_stl(args.output, name, stacked)


if __name__ == "__main__":
    main()
