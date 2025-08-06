from openCHA.tasks import BaseTask
from typing import Any, List
import pandas as pd
import os
from datetime import datetime, timedelta

class SleepDisordersIntervals(BaseTask):
    """
    Detects continuous intervals where each type of sleep disorder occurred, based on a given start time.
    """

    name: str = "sleep_disorders_intervals"
    chat_name: str = "SleepDisordersIntervals"
    description: str = (
        "Returns time intervals during which each sleep disorder (Obstructive Apnea, Central Apnea, "
        "Hypopnea) occurred continuously or semi-continuously throughout the night."
    )
    dependencies: List[str] = []
    inputs: List[str] = [
        "User ID in string (e.g., 'SID_001')",
        "Start time of collection in format 'HH:MM:SS' (e.g., '22:00:00')"
    ]
    outputs: List[str] = [
        "List of intervals (start, end, type) for each sleep disorder."
    ]
    output_type: bool = False
    return_direct: bool = False

    data_dir: str = "data"

    def _execute(self, inputs: List[Any]) -> Any:
        sid = inputs[0]
        start_time_str = inputs[1]

        try:
            collection_start_time = datetime.strptime(start_time_str, "%H:%M:%S")
        except ValueError:
            return f"Invalid start time format: '{start_time_str}'. Expected format is 'HH:MM:SS'."

        filename = f"{sid}_whole_df.csv"
        file_path = os.path.join(os.getcwd(), self.data_dir, f"data_64Hz", filename)

        try:
            df = pd.read_csv(file_path, usecols=[
                "TIMESTAMP", "Obstructive_Apnea", "Central_Apnea", "Hypopnea"
            ])
        except Exception as e:
            return f"Could not load sleep disorder data: {e}"

        disorder_cols = ["Obstructive_Apnea", "Central_Apnea", "Hypopnea"]
        for col in disorder_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        intervals = []
        gap_threshold = 30  # seconds: max gap between events to still consider it the same interval

        for disorder in disorder_cols:
            events = df[df[disorder] > 0]
            if events.empty:
                continue

            # convert TIMESTAMPs to real datetime
            times = (pd.to_timedelta(events["TIMESTAMP"], unit="s") + collection_start_time).sort_values()

            current_start = times.iloc[0]
            previous_time = times.iloc[0]

            for current_time in times.iloc[1:]:
                gap = (current_time - previous_time).total_seconds()
                if gap > gap_threshold:
                    # close previous interval
                    intervals.append({
                        "start": current_start.strftime("%H:%M:%S"),
                        "end": previous_time.strftime("%H:%M:%S"),
                        "type": disorder
                    })
                    # start new interval
                    current_start = current_time
                previous_time = current_time

            # add final interval
            intervals.append({
                "start": current_start.strftime("%H:%M:%S"),
                "end": previous_time.strftime("%H:%M:%S"),
                "type": disorder
            })

        if not intervals:
            return f"No sleep disorders detected for {sid}."

        return {
            "patient_id": sid,
            "event_type": "sleep_disorders_intervals",
            "interval_count": len(intervals),
            "intervals": intervals
        }
