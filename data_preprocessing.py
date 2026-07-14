# data_preprocessing.py

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import hashlib


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lower_snake_case.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def load_raw_matches(csv_path: str) -> pd.DataFrame:
    """
    Load the raw World Cup matches CSV file.
    """
    df = pd.read_csv(csv_path)
    df = standardize_column_names(df)
    return df


def add_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a unique integer match_id column.
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df["match_id"] = df.index.astype(int)
    return df


def map_stage_to_importance(stage: str) -> float:
    """
    Map the tournament stage to a numerical importance score.
    """
    if not isinstance(stage, str):
        return 1.0
    stage = stage.lower().strip()
 # You can adjust this mapping later if needed
    mapping = {
        "group": 1.0,
        "group stage": 1.0,
        "round of 16": 2.0,
        "quarter-finals": 3.0,
        "quarterfinals": 3.0,
        "quarter-final": 3.0,
        "semi-finals": 4.0,
        "semifinals": 4.0,
        "semi-final": 4.0,
        "third place": 4.0,
        "final": 5.0,
    }

    for key, value in mapping.items():
        if key in stage:
            return value

    return 1.0


def compute_match_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'importance_score' column based on the 'stage' column.
    """
    df = df.copy()
    stage_col_candidates = ["stage", "round", "match_stage"]

    stage_col = None
    for c in stage_col_candidates:
        if c in df.columns:
            stage_col = c
            break

    if stage_col is None:
        df["importance_score"] = 1.0
        return df

    df["importance_score"] = df[stage_col].apply(map_stage_to_importance)
    return df


def clean_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'attendance' column and create 'expected_attendance'.
    """
    df = df.copy()
    attendance_candidates = ["attendance", "attendence"]

    att_col = None
    for c in attendance_candidates:
        if c in df.columns:
            att_col = c
            break

    if att_col is None:
        df["expected_attendance"] = 40000.0
        return df

    df[att_col] = (
        df[att_col]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
    )

    df[att_col] = pd.to_numeric(df[att_col], errors="coerce")

    median_att = df[att_col].median()
    if np.isnan(median_att):
        median_att = 40000.0

    df["expected_attendance"] = df[att_col].fillna(median_att)

    return df


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most important columns for the assignment problem.
    """
    df = df.copy()

    cols_to_keep = []

    for c in [
        "match_id",
        "year",
        "datetime",
        "city",
        "stadium",
        "home_team_name",
        "away_team_name",
        "expected_attendance",
        "importance_score",
        "fan_base_city",
    ]:
        if c in df.columns:
            cols_to_keep.append(c)

    # Add a fallback for team names if they are named differently
    if "home_team_name" not in cols_to_keep:
        if "home_team" in df.columns:
            df["home_team_name"] = df["home_team"]
            cols_to_keep.append("home_team_name")

    if "away_team_name" not in cols_to_keep:
        if "away_team" in df.columns:
            df["away_team_name"] = df["away_team"]
            cols_to_keep.append("away_team_name")

    df = df[cols_to_keep]
    return df



def _deterministic_choice_from_weights(key: str,
                                       candidate_cities: List[str],
                                       weights: List[float]) -> str:
    """
    Deterministically choose one city based on a string key and a
    population-based weight distribution (no random module used).

    This makes the assignment:
    - reproducible (same match -> same city every run)
    - but still distributed according to city weights.
    """
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    cum_w = np.cumsum(w)

    # Hash the key to a pseudo-random number in [0, 1)
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Use first 8 hex digits -> 32-bit integer
    r = int(h[:8], 16) / float(16**8)

    # Find first cumulative weight >= r
    idx = np.searchsorted(cum_w, r, side="right")
    idx = min(idx, len(candidate_cities) - 1)
    return candidate_cities[idx]


def assign_fan_base_cities(
    df: pd.DataFrame,
    candidate_cities: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Assign a plausible Saudi fan base city for each match to enable travel modeling.

    - Uses a deterministic, population-based distribution instead of pure randomness.
    - Same match will always be assigned the same fan_base_city.
    - Larger cities (e.g. Riyadh, Jeddah) receive a greater share of matches.

    Methodology (for report):
    - We assume that the majority of fans for World Cup matches in Saudi Arabia
      will originate from major population centers.
    - We allocate matches to fan base cities using a population-weighted scheme,
      implemented via a deterministic hash of match attributes
      (year, home team, away team, match_id).
    """
    df = df.copy()

    if candidate_cities is None:
        candidate_cities = [
            "Riyadh",
            "Jeddah",
            "Dammam",
            "Mecca",
            "Medina",
            "Abha",
            "Al-Hasa",
        ]

    # Approximate population / demand weights (sum ≈ 1.0)
    if weights is None or len(weights) != len(candidate_cities):
        weights = [0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]

    def _assign_row(row: pd.Series) -> str:
        # If Saudi Arabia is playing, we can force a central fan base (e.g. Riyadh)
        for col in ["home_team_name", "away_team_name"]:
            if col in row and isinstance(row[col], str):
                if "saudi" in row[col].lower():
                    return "Riyadh"

        # Build a stable key from match attributes
        key_parts = []
        for col in ["year", "match_id", "home_team_name", "away_team_name"]:
            if col in row and pd.notna(row[col]):
                key_parts.append(str(row[col]))
        key = "|".join(key_parts)

        if not key:
            # Fallback if all attributes are missing
            key = "fallback_key"

        return _deterministic_choice_from_weights(key, candidate_cities, weights)

    df["fan_base_city"] = df.apply(_assign_row, axis=1)
    return df

def prepare_matches(
    csv_path: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Full pipeline to load and prepare the World Cup matches data.
    """
    df = load_raw_matches(csv_path)
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    df = add_match_id(df)
    df = compute_match_importance(df)
    df = clean_attendance(df)
    df = assign_fan_base_cities(df)
    df = select_relevant_columns(df)
    df = df.dropna(subset=["city", "stadium", "datetime"])
    df.to_csv("clean_matches.csv", index=False)
    
    return df