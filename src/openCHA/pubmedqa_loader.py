import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class PubMedQALoader:
    def __init__(self, json_path: str = None):
        if json_path is None:
            # Encontra o arquivo automaticamente a partir do diretório do script
            current_dir = Path(__file__).parent
            json_path = current_dir / "datasets" / "ori_pqal.json"

        self.json_path = Path(json_path)
        self.data = None

    def load(self) -> Dict[str, Any]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"✅ Carregado: {len(self.data)} questões")
        return self.data

    def get_subset(self, num_samples: int = 3) -> List[Dict[str, Any]]:
        if self.data is None:
            self.load()

        subset = []
        count = 0

        for doc_id, content in self.data.items():
            if count >= num_samples:
                break

            question = content.get("QUESTION")
            expected = content.get("final_decision")

            if question and expected:
                subset.append({
                    "id": doc_id,
                    "question": question,
                    "expected_answer": expected
                })
                count += 1

        return subset
