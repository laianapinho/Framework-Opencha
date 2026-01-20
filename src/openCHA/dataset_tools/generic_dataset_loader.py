"""
Generic Dataset Loader - Carrega e normaliza qualquer JSON com perguntas e respostas
"""
import json
import logging
import uuid
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from .dataset_detector import DatasetDetector

logger = logging.getLogger(__name__)


class GenericDatasetLoader:
    """Carrega e normaliza datasets em qualquer formato JSON"""

    def __init__(self, confidence_threshold: float = 0.80):
        """
        Args:
            confidence_threshold: Confiança mínima para auto-mapping
        """
        self.detector = DatasetDetector(confidence_threshold=confidence_threshold)
        self.data = None
        self.mapping = None
        self.dataset_type = None
        self.raw_data = None

    def load_from_json(self, json_content: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Carrega e tenta normalizar JSON arbitrário (suporta JSON e JSONL).

        Args:
            json_content: String contendo JSON ou JSONL (um objeto por linha)

        Returns:
            Tuple: (dados_normalizados, mapping_usado)
        """
        try:
            # Tenta parse como JSON primeiro
            try:
                parsed = json.loads(json_content)

                # Normaliza para lista
                if isinstance(parsed, dict):
                    # Se é dict com IDs como chaves (tipo PubMedQA)
                    data_list = []
                    for key, value in parsed.items():
                        if isinstance(value, dict):
                            # Adiciona o ID ao objeto
                            item = value.copy()
                            if 'id' not in item and '_id' not in item:
                                item['_original_id'] = key
                            data_list.append(item)
                        else:
                            # Skip se não for dict
                            continue
                elif isinstance(parsed, list):
                    data_list = parsed
                else:
                    raise ValueError("JSON deve ser um objeto ou array")

            except json.JSONDecodeError as e:
                # Se falhar, tenta parsear como objetos JSONL em múltiplas linhas
                # Detecta padrão: { ... } (pode ter quebras de linha no meio)
                logger.info("JSON inválido, tentando formato JSONL múltiplas linhas...")
                data_list = []
                valid_objects = 0

                try:
                    # Remove espaços em branco extremos e quebras de linha desnecessárias
                    # Mas mantém espaços dentro dos strings
                    content = json_content.strip()

                    # Estratégia: encontrar todos os objetos JSON válidos
                    # Procura por { ... } pareados
                    objects_json = []
                    brace_count = 0
                    current_obj = ""

                    for char in content:
                        current_obj += char

                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1

                            # Quando fecha um objeto
                            if brace_count == 0 and current_obj.count('{') > 0:
                                objects_json.append(current_obj.strip())
                                current_obj = ""

                    logger.info(f"Detectados {len(objects_json)} potenciais objetos JSON")

                    # Tenta fazer parse de cada objeto
                    for i, obj_str in enumerate(objects_json):
                        try:
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict) and obj:
                                data_list.append(obj)
                                valid_objects += 1
                        except (json.JSONDecodeError, ValueError):
                            continue

                    logger.info(f"Parsed: {valid_objects} objetos válidos de {len(objects_json)}")

                except Exception as parse_error:
                    logger.error(f"Erro ao fazer parse JSONL: {parse_error}")

                if not data_list:
                    raise ValueError(f"Não foi possível fazer parse do arquivo (tamanho: {len(json_content)} chars)")

            if not data_list:
                raise ValueError("Dataset vazio após parsing")

            logger.info(f"✅ Carregado JSON com {len(data_list)} items")

            # Armazena dados brutos
            self.raw_data = data_list

            # Detecta estrutura
            detection = self.detector.detect_structure(data_list)
            logger.info(f"📊 Detecção: Tipo={detection['dataset_type']}, Confiança={detection['overall_confidence']:.0%}")

            # Se confiança baixa, questiona o usuário
            if detection['needs_confirmation']:
                logger.warning(
                    f"⚠️  Confiança baixa na detecção ({detection['overall_confidence']:.0%})\n"
                    f"   Pergunta: {detection['question_field']} ({detection['question_confidence']:.0%})\n"
                    f"   Resposta: {detection['answer_field']} ({detection['answer_confidence']:.0%})"
                )

            # Cria mapping
            mapping = {
                'question_field': detection['question_field'],
                'answer_field': detection['answer_field'],
                'id_field': detection['id_field'],
                'dataset_type': detection['dataset_type'],
                'needs_confirmation': detection['needs_confirmation']
            }

            self.mapping = mapping
            self.dataset_type = detection['dataset_type']

            # Normaliza dados com o mapping
            normalized_data = self._normalize_data(data_list, mapping)

            self.data = normalized_data

            return normalized_data, mapping

        except json.JSONDecodeError as e:
            logger.error(f"❌ Erro ao fazer parse do JSON: {e}")
            raise ValueError(f"JSON inválido: {e}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset: {e}")
            raise

    def load_from_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Carrega JSON de arquivo.

        Args:
            file_path: Caminho do arquivo JSON

        Returns:
            Tuple: (dados_normalizados, mapping_usado)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.load_from_json(content)

    def apply_custom_mapping(
        self,
        data_list: Optional[List[Dict[str, Any]]] = None,
        mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Aplica um mapping customizado aos dados.

        Args:
            data_list: Lista de dados (usa self.raw_data se None)
            mapping: Dict com 'question_field', 'answer_field', 'id_field'

        Returns:
            Dados normalizados com novo mapping
        """
        if data_list is None:
            data_list = self.raw_data
        if mapping is None:
            raise ValueError("Mapping deve ser fornecido")

        # Valida mapping
        validation = self.detector.validate_mapping(data_list, mapping)
        if not validation['valid']:
            raise ValueError(f"Mapping inválido: {validation['errors']}")

        # Normaliza com novo mapping
        normalized = self._normalize_data(data_list, mapping)
        self.mapping = mapping
        self.data = normalized

        logger.info(f"✅ Mapping customizado aplicado: {mapping}")

        return normalized

    def _normalize_data(
        self,
        data_list: List[Dict[str, Any]],
        mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Normaliza dados para formato padrão interno.

        Args:
            data_list: Lista de items originais
            mapping: Mapeamento de campos

        Returns:
            Lista normalizada com campos: id, question, answer, original_item
        """
        normalized = []

        question_field = mapping.get('question_field')
        answer_field = mapping.get('answer_field')
        id_field = mapping.get('id_field')

        for i, item in enumerate(data_list):
            try:
                # Extrai pergunta
                question = str(item.get(question_field, '')).strip()
                if not question:
                    logger.warning(f"Item {i}: pergunta vazia, pulando")
                    continue

                # Extrai resposta
                answer = str(item.get(answer_field, '')).strip()
                if not answer:
                    logger.warning(f"Item {i}: resposta vazia, pulando")
                    continue

                # Extrai ou gera ID
                if id_field and id_field in item:
                    item_id = str(item[id_field])
                elif '_original_id' in item:
                    item_id = item['_original_id']
                else:
                    item_id = str(uuid.uuid4())

                normalized_item = {
                    'id': item_id,
                    'question': question,
                    'answer': answer,
                    'original_item': item  # Preserva item original
                }

                normalized.append(normalized_item)

            except Exception as e:
                logger.warning(f"Item {i}: erro ao normalizar - {e}")
                continue

        logger.info(f"✅ Normalizados {len(normalized)}/{len(data_list)} items")
        return normalized

    def get_subset(self, num_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Retorna um subset dos dados.

        Args:
            num_samples: Número de amostras

        Returns:
            Lista com questões formatadas para benchmark
        """
        if self.data is None:
            raise ValueError("Nenhum dataset carregado. Use load_from_json() ou load_from_file()")

        subset = self.data[:num_samples]

        # Formata para compatibilidade com benchmark
        formatted = []
        for item in subset:
            formatted.append({
                'id': item['id'],
                'question': item['question'],
                'expected_answer': item['answer'],
                'dataset_type': self.dataset_type
            })

        return formatted

    def get_all(self) -> List[Dict[str, Any]]:
        """Retorna todos os dados normalizados"""
        if self.data is None:
            raise ValueError("Nenhum dataset carregado")
        return self.data

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas sobre o dataset carregado"""
        if self.data is None:
            return {}

        answers = [item['answer'] for item in self.data]
        unique_answers = len(set(answers))

        avg_question_len = sum(len(item['question']) for item in self.data) / len(self.data)
        avg_answer_len = sum(len(item['answer']) for item in self.data) / len(self.data)

        return {
            'total_items': len(self.data),
            'unique_answers': unique_answers,
            'dataset_type': self.dataset_type,
            'mapping': self.mapping,
            'average_question_length': avg_question_len,
            'average_answer_length': avg_answer_len,
            'questions_by_length': {
                'short': sum(1 for item in self.data if len(item['question']) < 100),
                'medium': sum(1 for item in self.data if 100 <= len(item['question']) < 300),
                'long': sum(1 for item in self.data if len(item['question']) >= 300)
            }
        }

    def get_detection_result(self) -> Dict[str, Any]:
        """Retorna o resultado da detecção automática"""
        if self.raw_data is None:
            return {}

        return self.detector.detect_structure(self.raw_data)

    def get_mapping_suggestions(self) -> List[Dict[str, Any]]:
        """Retorna sugestões de mapping alternativas"""
        if self.raw_data is None:
            return []

        detection = self.detector.detect_structure(self.raw_data)
        return self.detector.generate_mapping_suggestions(detection)
