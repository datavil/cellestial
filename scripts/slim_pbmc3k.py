#!/usr/bin/env python
"""
Slim the preprocessed pbmc3k `.h5ad` by dropping the redundant `counts` layer.

The hosted `pbmc3k_pped.h5ad` stores the count matrix three times: `X`
(log-normalized), `.raw.X` (raw counts for all genes), and `layers['counts']`
(raw counts for the filtered genes). The `counts` layer is a gene-subset of
`.raw.X`, so removing it shrinks the file by roughly a third with nothing lost
that `.raw` does not already hold.

Usage
-----
    python scripts/slim_pbmc3k.py <input.h5ad> [output.h5ad]

If `output` is omitted, the input is overwritten in place.
"""

import sys
from pathlib import Path

from anndata import read_h5ad


def main() -> None:
    """Parse arguments, drop the `counts` layer, and write the result."""
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        raise SystemExit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) == 3 else input_file

    adata = read_h5ad(input_file)

    if "counts" not in adata.layers:
        print(f"No 'counts' layer in {input_file}; nothing to drop.")
        return

    del adata.layers["counts"]
    adata.write(output_file)

    before = input_file.stat().st_size / 1e6
    after = output_file.stat().st_size / 1e6
    print(f"Dropped layers['counts']: {before:.0f} MB -> {after:.0f} MB ({output_file})")


if __name__ == "__main__":
    main()
