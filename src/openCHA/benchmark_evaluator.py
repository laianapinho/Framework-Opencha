import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BenchmarkEvaluator:
    def __init__(self):
        self.yes_words = [
            'yes', 'sim', 'yep', 'true', 'verdadeiro', 'correto',
            'é verdade', 'é correto', 'com certeza', 'com certeza que sim',
            'play', 'papel', 'role', 'melhora', 'melhor', 'benefício',
            'benefícios', 'vantagem', 'vantagens', 'ajuda', 'ajudam',
            'previne', 'preveni', 'reduz', 'reduz', 'diminui', 'aumenta'
        ]

        self.no_words = [
            'no', 'não', 'nao', 'nope', 'false', 'falso',
            'não é verdade', 'não é correto', 'definitivamente não',
            'do not', 'does not', 'doesn\'t', 'don\'t',
            'não há', 'não tem', 'nenhum', 'nenhuma'
        ]

        self.maybe_words = [
            'maybe', 'talvez', 'perhaps', 'possibly', 'pode',
            'might', 'could', 'uncertain', 'incerto', 'desconhecido',
            'unclear', 'unclear', 'seems', 'appears', 'suggest',
            'sugere', 'possível', 'é possível', 'pode ser'
        ]

    def extract_answer(self, text: str) -> str:
        """
        Extrai yes/no/maybe do texto longo.
        Procura principalmente no início do texto.
        """
        if not text:
            return "unknown"

        text_lower = text.lower()

        # Pega os primeiros 300 caracteres para análise principal
        text_start = text_lower[:300]

        # Conta ocorrências em todo o texto mas pesa mais o início
        yes_count_start = sum(1 for w in self.yes_words if w in text_start) * 3
        no_count_start = sum(1 for w in self.no_words if w in text_start) * 3
        maybe_count_start = sum(1 for w in self.maybe_words if w in text_start) * 3

        # Conta no texto completo
        yes_count_full = sum(1 for w in self.yes_words if w in text_lower)
        no_count_full = sum(1 for w in self.no_words if w in text_lower)
        maybe_count_full = sum(1 for w in self.maybe_words if w in text_lower)

        # Combina (início tem peso maior)
        yes_count = yes_count_start + (yes_count_full - sum(1 for w in self.yes_words if w in text_start))
        no_count = no_count_start + (no_count_full - sum(1 for w in self.no_words if w in text_start))
        maybe_count = maybe_count_start + (maybe_count_full - sum(1 for w in self.maybe_words if w in text_start))

        logger.debug(f"Contagem: yes={yes_count}, no={no_count}, maybe={maybe_count}")

        # Decide baseado na contagem
        if yes_count > no_count and yes_count > maybe_count and yes_count > 0:
            return "yes"
        elif no_count > yes_count and no_count > maybe_count and no_count > 0:
            return "no"
        elif maybe_count > 0:
            return "maybe"

        return "unknown"

    def evaluate(self, expected: str, model_response: str) -> Dict[str, Any]:
        """
        Compara resposta esperada com resposta do modelo.

        Args:
            expected: Resposta correta (yes/no/maybe)
            model_response: Resposta completa do modelo

        Returns:
            Dict com resultado da avaliação
        """
        extracted = self.extract_answer(model_response)
        correct = extracted == expected.lower()

        logger.debug(f"Avaliação: esperado={expected}, extraído={extracted}, correto={correct}")

        return {
            "correct": correct,
            "expected": expected.lower(),
            "extracted": extracted
        }
