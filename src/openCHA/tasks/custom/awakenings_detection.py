from openCHA.tasks import BaseTask
from typing import Any, List
import pandas as pd
import os

class AwakeningsDetection(BaseTask):
    """
    **Description:**
        This task detects and counts the number of awakenings during the participant's sleep using the 'Sleep_Stage' column.
        It analyzes transitions from any non-wake stage (N1, N2, N3, R) to wake stage (W) and returns the number of discrete wake episodes.
        Consecutive 'W' entries are grouped as one awakening.
    """

    name: str = "awakenings_detection"
    chat_name: str = "AwakeningsDetection"
    description: str = (
        "Counts how many times the participant transitioned into wakefulness (Sleep_Stage = 'W') during the night. "
        "Only transitions from a sleep stage to 'W' are considered awakenings. "
        "Multiple consecutive 'W' values are treated as a single awakening episode."
    )
    dependencies: List[str] = []
    inputs: List[str] = [
        "User ID in string (e.g., 'SID_001')",
        "Data frequency version: '64Hz' or '100Hz'"
    ]
    outputs: List[str] = [
        "Total number of awakening episodes during the night in integer format."
    ]
    output_type: bool = False
    return_direct: bool = False

    data_dir: str = "data"

    def _execute(self, inputs: List[Any]) -> str:
        sid = inputs[0]
        frequency = inputs[1]

        if frequency not in ['64Hz', '100Hz']:
            return "Invalid frequency. Use '64Hz' or '100Hz'."

        filename = f"{sid}_whole_df.csv" if frequency == "64Hz" else f"{sid}_PSG_df.csv"
        file_path = os.path.join(os.getcwd(), self.data_dir, f"data_{frequency}", filename)

        try:
            df = pd.read_csv(file_path, usecols=["TIMESTAMP", "Sleep_Stage"])
        except Exception as e:
            return f"Could not load sleep data: {e}"

        if 'Sleep_Stage' not in df.columns:
            return "Sleep_Stage column is missing."

        sleep_stage = df['Sleep_Stage'].fillna(method='ffill').astype(str).str.strip()
        INVALID_STAGES = ['Missing', 'P']
        valid_stages = sleep_stage[~sleep_stage.isin(INVALID_STAGES)]

        if valid_stages.empty:
            return "No valid sleep stage data available."

        stage_shift = valid_stages.shift(1)
        awakenings = ((valid_stages == 'W') & (stage_shift != 'W')).sum()

        return f"total: {awakenings} awakenings"

