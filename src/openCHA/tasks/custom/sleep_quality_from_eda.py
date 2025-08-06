from openCHA.tasks import BaseTask
from typing import Any, List, Union, Dict
import os
import pandas as pd

class SleepQualityFromEDA(BaseTask):
    """
    **Description:**
        This task estimates a coarse binary sleep quality ("good" or "poor") based solely on EDA features.
        It uses simple threshold rules on the number of EDA epochs and EDA storms extracted by `EDAFeatureExtraction`.
        Intended as a baseline method without use of machine learning.
    """

    name: str = "sleep_quality_from_eda"
    chat_name: str = "SleepQualityFromEDA"
    description: str = (
        "Estimates binary sleep quality ('good' or 'poor') based on EDA features.\n"
        "Uses threshold rules on number of EDA epochs and EDA storms to determine quality."
    )
    dependencies: List[str] = []
    inputs: List[str] = [
        "User ID in string (e.g., 'SID_001')"
    ]
    outputs: List[str] = [
        "'good' or 'poor' sleep quality estimated from EDA features."
    ]
    output_type: bool = False
    return_direct: bool = False

    features_dir: str = "outputs"

    def _execute(self, inputs: List[Any]) -> Union[str, Dict[str, str]]:
        sid = inputs[0]

        # Caminho esperado do arquivo de features gerado pela EDAFeatureExtraction
        file_path = os.path.join(os.getcwd(), self.features_dir, f"{sid}_eda_features.json")

        try:
            eda_data = pd.read_json(file_path)
        except Exception as e:
            return f"Could not load EDA features: {e}"

        if "eda_features" not in eda_data.columns:
            return "Missing EDA features in input file."

        # Extrai o dicionário com os dados de features
        features = eda_data["eda_features"].iloc[0]

        # Heurística simples para classificação binária da qualidade do sono
        epochs = features.get("number_of_eda_epochs", 0)
        storms = features.get("number_of_eda_storms", 0)

        if epochs >= 30 and storms >= 5:
            quality = "good"
        else:
            quality = "poor"

        return {
            "patient_id": sid,
            "sleep_quality": quality
        }
