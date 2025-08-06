from openCHA.tasks import BaseTask
from typing import Any, List, Dict, Union
import pandas as pd
import numpy as np
import os


class EDAFeatureExtraction(BaseTask):
    """
    **Description:**
        This task extracts features from the Empatica E4's Electrodermal Activity (EDA) signal during the participant's sleep.
        It computes metrics like:
        - Number of EDA events (delta > 0.01 μS/s)
        - Number of EDA epochs (30s windows containing at least one event)
        - Number of EDA storms (2 or more consecutive epochs)
        - Statistics on storm sizes: average, standard deviation, and maximum length
    """

    name: str = "eda_feature_extraction"
    chat_name: str = "EDAFeatureExtraction"
    description: str = (
        "Extracts EDA signal features during the night:\n"
        "- Number of EDA events (delta > 0.01 μS/s)\n"
        "- Number of 30-second epochs containing EDA activity\n"
        "- Number and size of EDA 'storms' (2+ consecutive active epochs)"
    )
    dependencies: List[str] = []
    inputs: List[str] = [
        "User ID in string (e.g., 'SID_001')"
    ]
    outputs: List[str] = [
        "Dictionary with extracted EDA features for the night."
    ]
    output_type: bool = False
    return_direct: bool = False

    data_dir: str = "data"

    def _execute(self, inputs: List[Any]) -> Union[str, Dict[str, Any]]:
        sid = inputs[0]
        file_path = os.path.join(os.getcwd(), self.data_dir, "data_64Hz", f"{sid}_whole_df.csv")

        try:
            df = pd.read_csv(file_path, usecols=["TIMESTAMP", "EDA"])
        except Exception as e:
            return f"Could not load EDA data: {e}"

        df["EDA"] = pd.to_numeric(df["EDA"], errors="coerce").interpolate().fillna(method='bfill')

        # Compute derivative of EDA signal
        df["delta"] = df["EDA"].diff() / df["TIMESTAMP"].diff()
        df["delta"].fillna(0, inplace=True)

        # EDA events: delta > 0.01 μS/s
        event_threshold = 0.01
        df["event"] = df["delta"] > event_threshold
        event_times = df[df["event"]]["TIMESTAMP"].values

        # Epochs: 30-second windows with ≥1 EDA event
        epoch_duration = 30  # in seconds
        max_time = df["TIMESTAMP"].max()
        num_epochs = int(np.ceil(max_time / epoch_duration))
        epochs = [False] * num_epochs

        for t in event_times:
            idx = int(t // epoch_duration)
            if idx < len(epochs):
                epochs[idx] = True

        # Storms: 2 or more consecutive active epochs
        storms = []
        current_storm = []

        for is_active in epochs:
            if is_active:
                current_storm.append(1)
            else:
                if len(current_storm) >= 2:
                    storms.append(current_storm)
                current_storm = []
        if len(current_storm) >= 2:
            storms.append(current_storm)

        storm_lengths = [len(s) for s in storms]

        features = {
            "number_of_eda_events": int(df["event"].sum()),
            "number_of_eda_epochs": int(sum(epochs)),
            "number_of_eda_storms": len(storms),
            "average_storm_size": round(np.mean(storm_lengths), 2) if storm_lengths else 0,
            "std_storm_size": round(np.std(storm_lengths), 2) if storm_lengths else 0,
            "largest_storm_size": max(storm_lengths) if storm_lengths else 0
        }

        return {
            "patient_id": sid,
            "eda_features": features
        }
