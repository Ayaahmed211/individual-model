"""
Quick diagnostic: print every distinct DXTRT value actually found in DX.xpt,
with a count of how many records have it, plus a repr() so we can see
hidden whitespace/encoding issues that a normal print() would hide.
"""
import os
import pandas as pd

RAW_DATA_DIR = r"D:\T1DEXI - DATA FOR UPLOAD\T1DEXI - DATA FOR UPLOAD"
FILE_PATH_OVERRIDES = {
    "FAMLPM.xpt": r"C:\Users\3com\OneDrive\Desktop\FAMLPM.xpt",
}


def resolve_path(filename):
    return FILE_PATH_OVERRIDES.get(filename, os.path.join(RAW_DATA_DIR, filename))


def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


path = resolve_path("DX.xpt")
counts = {}
reader = pd.read_sas(path, format="xport", chunksize=200_000)
for chunk in reader:
    if "DXTRT" not in chunk.columns:
        print("DXTRT column not found! Columns are:", chunk.columns.tolist())
        break
    vals = chunk["DXTRT"].apply(decode)
    for v in vals:
        counts[v] = counts.get(v, 0) + 1

print(f"\nFound {len(counts)} distinct DXTRT values:\n")
for v, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  count={c:5d}  repr={v!r}")