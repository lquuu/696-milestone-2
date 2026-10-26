"""
Rebuild DS08a correctly from the original N-CMAPSS HDF5 file. Streaming, with one engine at a time.

In the HDF5 file, W (operating conditions), X_s (sensors), X_v (virtual sensors), T (health parameters / *_mod), Y (RUL) & A (unit, cycle, Fc, hs) are SEPARATE arrays with the SAME number of rows, already aligned row-by-row.

Row 'i' of every array is the same second of the same flight, so we place them side by side (column-stack). No join on unit/cycle, and nothing gets dropped or repeated.

OUTPUTS OF THIS FILE (Parquet format, for easy Pandas reading)
  1. ds08a_rows.parquet    — Downsampled row-level table (all columns, incl. T and RUL) strides: Fc1 = 76 s, Fc2 = 127 s, Fc3 = 180 s (your current file)
  
  2. ds08a_flights.parquet — 1 row per (unit, cycle) computed from the FULL 1 Hz data:
                             mean / std / q10 / q50 / q90 of W, X_s, X_v; mean of T; first of A
  
  3. ds08a_units.parquet   — 1 row per engine: split (dev/test), Fc, n_cycles,
                             first unhealthy cycle, and;
                             per-component degradation at end of life (from T) --> Use to derive Part A component / E-vs-F labels
"""

import sys, os
import numpy as np
import pandas as pd
import h5py

STRIDE = {1: 76, 2: 127, 3: 180}           # match the current downsampled file
QUANTS = [0.10, 0.50, 0.90]
GROUPS = ["W", "X_s", "X_v", "T", "Y", "A"]


def var_names(f, g):
    key = f"{g}_var"
    if key in f:
        return [n.decode() if isinstance(n, bytes) else str(n) for n in np.array(f[key]).ravel()]
    if g == "Y":
        return ["RUL"]
    raise KeyError(key)


def unit_slices(A_unit):
    """Rows are stored contiguously per unit -> return {unit: (start, stop)}."""
    units, starts = np.unique(A_unit, return_index=True)
    order = np.argsort(starts)
    units, starts = units[order], starts[order]
    stops = list(starts[1:]) + [len(A_unit)]
    # sanity check: contiguous blocks
    assert np.all(np.diff(starts) > 0)
    return {float(u): (int(s), int(e)) for u, s, e in zip(units, starts, stops)}


def main(h5_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows_out, flights_out, units_out = [], [], []

    with h5py.File(h5_path, "r") as f:
        names = {g: var_names(f, g) for g in GROUPS}
        print({g: len(v) for g, v in names.items()})
        a_cols = names["A"]                       # expected: unit, cycle, Fc, hs
        iu = a_cols.index("unit")

        for split in ["dev", "test"]:
            A_all_unit = np.array(f[f"A_{split}"][:, iu])   # one small column only
            n_rows = len(A_all_unit)
            for g in GROUPS:                                 # verify alignment
                assert f[f"{g}_{split}"].shape[0] == n_rows, f"{g}_{split} row count mismatch"

            for unit, (s, e) in unit_slices(A_all_unit).items():
                parts = []
                for g in GROUPS:
                    arr = np.asarray(f[f"{g}_{split}"][s:e], dtype=np.float32)
                    parts.append(pd.DataFrame(arr.reshape(e - s, -1), columns=names[g]))
                d = pd.concat(parts, axis=1)
                d.insert(0, "raw_idx", np.arange(s, e))     # position within the split
                d.insert(0, "split", split)
                d[["unit", "cycle", "Fc", "hs"]] = d[["unit", "cycle", "Fc", "hs"]].astype(int)
                d["EPR"] = d["P50"] / d["P2"]

                fc = int(d["Fc"].iloc[0])
                print(f"{split} unit {unit:.0f}: {e - s:,} rows, Fc={fc}, cycles={d.cycle.max()}")

                # 1) downsampled rows
                rows_out.append(d.iloc[:: STRIDE[fc]].copy())

                # 2) per-flight features from the full 1 Hz data
                feat_cols = names["W"] + names["X_s"] + names["X_v"] + ["EPR"]
                g = d.groupby("cycle")
                stats = [g[feat_cols].mean().add_suffix("_mean"),
                         g[feat_cols].std().add_suffix("_std")]
                for q in QUANTS:
                    stats.append(g[feat_cols].quantile(q).add_suffix(f"_q{int(q*100)}"))
                stats.append(g[names["T"]].mean().add_suffix("_flightmean"))  # labels/eval only!
                stats.append(g[["RUL"]].min())
                meta = g[["unit", "Fc", "hs"]].first()
                meta["n_seconds"] = g.size()
                fl = pd.concat([meta] + stats, axis=1).reset_index()
                fl.insert(0, "split", split)
                flights_out.append(fl)

                # 3) per-engine summary (degradation at end of life, from T)
                last = d[d.cycle == d.cycle.max()][names["T"]].mean()
                first = d[d.cycle == d.cycle.min()][names["T"]].mean()
                u = {"split": split, "unit": int(unit), "Fc": fc,
                     "n_cycles": int(d.cycle.max()),
                     "first_unhealthy_cycle": int(d.loc[d.hs == 0, "cycle"].min()) if (d.hs == 0).any() else None}
                u.update({f"{c}_delta": float(last[c] - first[c]) for c in names["T"]})
                units_out.append(u)
                del d, parts

    pd.concat(rows_out).to_parquet(os.path.join(out_dir, "ds08a_rows.parquet"), index=False)
    pd.concat(flights_out).to_parquet(os.path.join(out_dir, "ds08a_flights.parquet"), index=False)
    pd.DataFrame(units_out).to_parquet(os.path.join(out_dir, "ds08a_units.parquet"), index=False)
    print("done ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
