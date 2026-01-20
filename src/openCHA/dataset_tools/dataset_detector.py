"""
Dataset Detector - Detecta tipo de dataset (fechado/aberto) e mapeia campos automaticamente
"""
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import Counter
import re

logger = logging.getLogger(__name__)


class DatasetDetector:
    """Detecta tipo de dataset e tenta fazer mapping automático de campos"""

    # Palavras-chave para detectar pergunta
    QUESTION_KEYWORDS = {
        'question', 'pergunta', 'query', 'consulta', 'prompt',
        'text', 'texto', 'input', 'entrada', 'problem', 'problema',
        'enunciado', 'ask', 'pedido', 'clinical_case', 'caso_clinico'
    }

    # Palavras-chave para detectar resposta
    ANSWER_KEYWORDS = {
        'answer', 'resposta', 'expected_answer', 'resposta_esperada',
        'label', 'label_answer', 'gold_label', 'gold_answer',
        'final_decision', 'decision', 'decisao', 'resultado', 'result',
        'output', 'saida', 'target', 'alvo', 'ground_truth'
    }

    # Palavras-chave para detectar ID
    ID_KEYWORDS = {
        'id', 'doc_id', 'question_id', 'unique_id', 'exam_id',
        'sample_id', 'instance_id', 'index'
    }

    def __init__(self, confidence_threshold: float = 0.80):
        """
        Args:
            confidence_threshold: Confiança mínima para auto-mapping (0-1)
        """
        self.confidence_threshold = confidence_threshold

    def detect_dataset_type(self, data: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Detecta se dataset é fechado (classificação) ou aberto (geração).

        Args:
            data: Lista de items do dataset

        Returns:
            Tuple: (tipo, confianca) onde tipo é 'closed' ou 'open'
        """
        if not data:
            return 'closed', 0.5

        try:
            # Tenta encontrar o campo de resposta
            answer_field = self._find_answer_field(data)
            if not answer_field:
                return 'open', 0.5

            # Coleta todas as respostas
            answers = []
            for item in data:
                if answer_field in item:
                    answer = str(item[answer_field]).strip()
                    if answer:
                        answers.append(answer)

            if not answers:
                return 'open', 0.5

            # Calcula estatísticas
            unique_answers = len(set(answers))
            total_answers = len(answers)
            avg_length = sum(len(a) for a in answers) / len(answers) if answers else 0

            # Heurística melhorada:
            # Se respostas são longas (>10 chars) = aberta
            # Se poucos valores únicos (<= 3) E respostas curtas (<=3 chars) = fechada

            if avg_length > 10:
                # Respostas longas = dataset aberto
                confidence = 0.95 if avg_length > 50 else 0.85
                return 'open', confidence

            elif unique_answers <= 3:
                # Poucos valores únicos e respostas curtas = dataset fechado
                confidence = 0.95 if unique_answers <= 2 else 0.85
                return 'closed', confidence

            else:
                # Caso intermediário - provavelmente aberto
                return 'open', 0.70

        except Exception as e:
            logger.warning(f"Erro ao detectar tipo: {e}")
            return 'open', 0.5

    def detect_structure(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detecta estrutura do JSON e tenta fazer mapping automático.

        Args:
            data: Lista de items do dataset

        Returns:
            Dict com campos detectados e confiança
        """
        if not data:
            return {
                'question_field': None,
                'answer_field': None,
                'id_field': None,
                'confidence': 0.0,
                'all_fields': [],
                'needs_confirmation': True
            }

        first_item = data[0]
        all_fields = list(first_item.keys())

        # Detecta cada campo
        question_field, q_conf = self._find_field(first_item, self.QUESTION_KEYWORDS)
        answer_field, a_conf = self._find_field(first_item, self.ANSWER_KEYWORDS)
        id_field, id_conf = self._find_field(first_item, self.ID_KEYWORDS)

        # Confiança geral é a média dos campos principais
        main_confidence = (q_conf + a_conf) / 2

        result = {
            'question_field': question_field,
            'question_confidence': q_conf,
            'answer_field': answer_field,
            'answer_confidence': a_conf,
            'id_field': id_field,
            'id_confidence': id_conf,
            'overall_confidence': main_confidence,
            'all_fields': all_fields,
            'needs_confirmation': main_confidence < self.confidence_threshold,
            'dataset_type': self.detect_dataset_type(data)[0],
            'dataset_type_confidence': self.detect_dataset_type(data)[1]
        }

        return result

    def _find_field(self, item: Dict[str, Any], keywords: set) -> Tuple[Optional[str], float]:
        """
        Encontra campo que melhor corresponde aos keywords.

        Returns:
            Tuple: (field_name, confidence)
        """
        matches = {}

        for field_name in item.keys():
            field_lower = field_name.lower()

            # Match exato
            if field_lower in keywords:
                matches[field_name] = 1.0
            # Match parcial (contains)
            elif any(kw in field_lower for kw in keywords):
                matches[field_name] = 0.8
            # Match com underscore/camelCase
            elif any(kw in field_lower.replace('_', '').replace('-', '') for kw in keywords):
                matches[field_name] = 0.6

        if not matches:
            return None, 0.0

        # Retorna o melhor match
        best_field = max(matches, key=matches.get)
        return best_field, matches[best_field]

    def _find_answer_field(self, data: List[Dict[str, Any]]) -> Optional[str]:
        """Encontra o campo de resposta para análise de tipo"""
        if not data:
            return None

        first_item = data[0]
        field, _ = self._find_field(first_item, self.ANSWER_KEYWORDS)
        return field

    def validate_mapping(self, data: List[Dict[str, Any]], mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Valida se o mapping é válido para os dados.

        Args:
            data: Lista de items
            mapping: Dict com 'question_field', 'answer_field', 'id_field'

        Returns:
            Dict com validação e mensagens
        """
        errors = []
        warnings = []

        if not data:
            errors.append("Dataset vazio")
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        first_item = data[0]

        # Valida question_field
        if not mapping.get('question_field'):
            errors.append("Campo de pergunta não especificado")
        elif mapping['question_field'] not in first_item:
            errors.append(f"Campo de pergunta '{mapping['question_field']}' não existe")

        # Valida answer_field
        if not mapping.get('answer_field'):
            errors.append("Campo de resposta não especificado")
        elif mapping['answer_field'] not in first_item:
            errors.append(f"Campo de resposta '{mapping['answer_field']}' não existe")

        # Valida id_field (opcional)
        if mapping.get('id_field') and mapping['id_field'] not in first_item:
            warnings.append(f"Campo de ID '{mapping['id_field']}' não existe, será gerado automaticamente")

        # Checa se todos os items têm os campos obrigatórios
        for i, item in enumerate(data):
            if mapping.get('question_field') not in item or not str(item[mapping['question_field']]).strip():
                errors.append(f"Item {i}: campo de pergunta vazio ou ausente")
            if mapping.get('answer_field') not in item or not str(item[mapping['answer_field']]).strip():
                errors.append(f"Item {i}: campo de resposta vazio ou ausente")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'items_checked': len(data)
        }

    def generate_mapping_suggestions(self, detection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gera sugestões de mapping alternativas baseado na detecção.

        Args:
            detection_result: Resultado de detect_structure()

        Returns:
            Lista de sugestões alternativas
        """
        suggestions = []

        # Sugestão 1: Mapping principal (melhor match)
        if detection_result['overall_confidence'] >= self.confidence_threshold:
            suggestions.append({
                'rank': 1,
                'confidence': detection_result['overall_confidence'],
                'mapping': {
                    'question_field': detection_result['question_field'],
                    'answer_field': detection_result['answer_field'],
                    'id_field': detection_result['id_field']
                },
                'reason': 'Melhor match automático'
            })

        # Sugestões adicionais com campos alternativos
        all_fields = detection_result['all_fields']
        if len(all_fields) > 2:
            # Tenta outras combinações
            question_candidates = [
                f for f in all_fields
                if self._find_field({'field': None}, self.QUESTION_KEYWORDS)[1] > 0.3
                or f != detection_result['question_field']
            ]

            for alt_question in question_candidates[:2]:
                if alt_question != detection_result['question_field']:
                    suggestions.append({
                        'rank': len(suggestions) + 1,
                        'confidence': 0.5,
                        'mapping': {
                            'question_field': alt_question,
                            'answer_field': detection_result['answer_field'],
                            'id_field': detection_result['id_field']
                        },
                        'reason': f"Alternativa: usando campo '{alt_question}' para pergunta"
                    })

        return suggestions[:3]  # Retorna top 3 sugestões
