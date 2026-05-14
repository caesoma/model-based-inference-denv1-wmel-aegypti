import pandas as pd
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from denv_wmel.preprocessing import preprocess


def test_preprocess_runs():
    df = preprocess()
    assert len(df) > 0
    assert set(["group", "dpi", "dose_exp", "denv_titer", "wmel_titer"]).issubset(df.columns)


def test_min_denv_titer_is_one():
    df = preprocess()
    pos = df[df["denv_titer"] > 0]["denv_titer"]
    assert pos.min() == 1


def test_min_wmel_titer_is_one():
    df = preprocess()
    br = df[~df["group"].str.upper().str.contains("TET")]
    pos = br[br["wmel_titer"] > 0]["wmel_titer"]
    assert pos.min() == 1


def test_groups_present():
    df = preprocess()
    groups = df["group"].unique()
    has_tet  = any("TET"  in g.upper() for g in groups)
    has_wmel = any("MEL"  in g.upper() or "BR" in g.upper() for g in groups)
    assert has_tet and has_wmel


def test_correct_doses():
    df = preprocess()
    assert set(df["dose_exp"].unique()) == {4, 5, 6, 7, 8}


def test_correct_timepoints():
    df = preprocess()
    assert set(df["dpi"].unique()) == {3, 7, 14}
