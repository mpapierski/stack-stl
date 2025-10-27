# Vertical STL Stacking

`stack_stl.py` stacks one or more STL models vertically so you can print several copies in a single job. Provide one or more input meshes followed by the output path and the script will normalize each mesh, align them along Z, and generate a stacked STL with an optional gap between parts.

## Usage

```bash
python stack_stl.py INPUT.stl [ADDITIONAL_INPUT.stl ...] OUTPUT.stl [options]
```

- Pass one or more input STLs and finish with the destination file.
- When multiple inputs are given they are first stacked (with the configured gap) and treated as a single “base stack”.
- The base stack is then duplicated `--clones` times, again respecting the chosen gap between copies.

### Common Examples

```bash
# Stack a single model five times, 0.3 mm apart, centered on XY
python stack_stl.py part.stl output.stl --clones 5 --gap 0.3 --center

# Combine two models, repeat the pair three times, and force binary output
python stack_stl.py a.stl b.stl combo.stl --clones 3 --gap 0.4 --output-format binary
```

## Options

- `--clones`, `-c`: Total number of vertically stacked copies. `1` keeps the base stack only.
- `--gap`, `-g`: Vertical spacing (in model units) between each stacked copy. In my testing, a `--gap` of `0.5` works best for a `0.2` mm layer height.
- `--center`: Center every input around the XY origin before stacking. Helpful when input meshes are offset.
- `--output-format`: Choose `ascii`, `binary`, or `auto` (default). `auto` mirrors the format of the first input mesh when possible.

## Output

- The output file inherits the name of the first STL when written as ASCII; binary exports reuse the original header when available.
- The script creates parent folders for the output path if they do not exist.
- Invalid or empty STLs raise an error; ensure inputs are valid meshes.

## Slicer Recommendations

To keep the stacked prints easy to separate, configure your slicer with:

1. **Support Style: Snug** – for flat models this dramatically reduces print time (YMMV).
2. **Support Interface: PLA/PETG** – if your main part uses PLA, run PETG for the interface (swap materials accordingly for other combinations).

These settings have produced consistent, easily separable stacks alongside the gap guidance above.

## Author & License

Created and maintained by [Michał Papierski](michal@papierski.net). The project is released under the MIT License, allowing you to use, modify, and distribute the code with proper attribution.
