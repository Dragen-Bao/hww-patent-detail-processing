"""Core quarterly KPST formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_field_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Quarter-specific IPC-field adjustment × FS/BS."""
    out = frame.copy()
    out["field_adjustment"] = np.nan

    for _, quarter in out.groupby("application_quarter"):
        total_bs = quarter["bs"].sum()
        if total_bs == 0:
            continue
        total_n = len(quarter)
        for _, field in quarter.groupby("ipc_main_group", dropna=False):
            field_bs = field["bs"].sum()
            xi = (total_n / len(field)) * (field_bs / total_bs)
            out.loc[field.index, "field_adjustment"] = xi

    out["kpst_quality"] = out["field_adjustment"] * (out["fs"] / out["bs"])
    return out
