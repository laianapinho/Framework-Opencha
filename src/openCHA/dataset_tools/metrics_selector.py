"""
Metrics Selector - Seleciona métricas apropriadas baseado no tipo de dataset
"""
import logging
from typing import Dict, List, Any, Callable
from enum import Enum

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

# Para métricas de texto
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score não instalado. Instale com: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  nltk não instalado. Instale com: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️  sentence-transformers não instalado. Instale com: pip install sentence-transformers")

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Tenta baixar WordNet automaticamente
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Baixando WordNet...")
        nltk.download('wordnet', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    logger.warning("METEOR não disponível")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠️  bert-score não instalado. Instale com: pip install bert-score")


class DatasetType(Enum):
    """Tipos de dataset"""
    CLOSED = "closed"  # Classificação (yes/no/maybe, labels, etc)
    OPEN = "open"  # Geração (respostas em texto livre)


class MetricsSelector:
    """Seleciona e calcula métricas apropriadas para o tipo de dataset"""

    def __init__(self):
        self.embedding_model = None
        self._embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def get_metrics_for_type(self, dataset_type: str) -> Dict[str, str]:
        """
        Retorna lista de métricas recomendadas para o tipo de dataset.

        Args:
            dataset_type: 'closed' ou 'open'

        Returns:
            Dict com métricas disponíveis
        """
        if dataset_type == DatasetType.CLOSED.value or dataset_type == 'closed':
            return {
                'accuracy': 'Acurácia - proporção de predições corretas',
                'precision': 'Precisão - proporção de predições positivas corretas',
                'recall': 'Recall - proporção de positivos identificados',
                'f1': 'F1-Score - média harmônica entre precisão e recall',
                'confusion_matrix': 'Matriz de confusão - distribuição de predições'
            }
        else:  # open
            return {
                'bleu': 'BLEU - Precisão de n-gramas (1-4)',
                'rouge_l': 'ROUGE-L - Longest common subsequence F-score',
                'semantic_similarity': 'Similaridade semântica - embedding-based',
            }

    def calculate_closed_metrics(
        self,
        y_true: List[str],
        y_pred: List[str]
    ) -> Dict[str, Any]:
        """
        Calcula métricas para dataset fechado (classificação).

        Args:
            y_true: Lista de rótulos esperados
            y_pred: Lista de rótulos preditos

        Returns:
            Dict com todas as métricas
        """
        try:
            # Normaliza strings
            y_true_norm = [str(y).strip().lower() for y in y_true]
            y_pred_norm = [str(y).strip().lower() for y in y_pred]

            metrics = {
                'accuracy': accuracy_score(y_true_norm, y_pred_norm),
                'precision': precision_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0),
                'recall': recall_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0),
                'f1': f1_score(y_true_norm, y_pred_norm, average='weighted', zero_division=0),
            }

            # Tenta adicionar confusion matrix e classification report
            try:
                unique_labels = sorted(set(y_true_norm) | set(y_pred_norm))
                metrics['confusion_matrix'] = confusion_matrix(
                    y_true_norm, y_pred_norm, labels=unique_labels
                ).tolist()
                metrics['labels'] = unique_labels

                metrics['classification_report'] = classification_report(
                    y_true_norm, y_pred_norm, zero_division=0, output_dict=True
                )
            except Exception as e:
                logger.warning(f"Erro ao calcular confusion matrix: {e}")

            logger.info(f"✅ Métricas de classificação calculadas")
            return metrics

        except Exception as e:
            logger.error(f"❌ Erro ao calcular métricas de classificação: {e}")
            raise

    def calculate_open_metrics(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Calcula métricas para dataset aberto (geração de texto).

        Args:
            references: Lista de respostas esperadas (gold standard)
            predictions: Lista de predições dos modelos

        Returns:
            Dict com todas as métricas
        """
        logger.info(f"🔍 DEBUG: Iniciando calculate_open_metrics com {len(references)} refs e {len(predictions)} preds")

        if len(references) != len(predictions):
            raise ValueError("Referências e predições devem ter o mesmo tamanho")

        metrics = {
            'count': len(references),
            'individual_scores': []
        }

        logger.info(f"🔍 DEBUG: BLEU_AVAILABLE={BLEU_AVAILABLE}, METEOR_AVAILABLE={METEOR_AVAILABLE}, BERTSCORE_AVAILABLE={BERTSCORE_AVAILABLE}")

        # ============================================================================
        # MÉTRICA 1: BLEU (Bilingual Evaluation Understudy)
        # ============================================================================
        # O que é: Avalia a qualidade da geração de texto comparando n-gramas
        #          (sequências de 1, 2, 3, 4 palavras) entre a predição e referência
        #
        # Como funciona:
        # - Quebra texto em tokens (palavras)
        # - Compara n-gramas de tamanhos 1-4 entre predição e referência
        # - Score de 0 a 1 (1 = match perfeito)
        # - Usa smoothing para evitar score 0 quando não há matches
        #
        # Vantagem: Rápido, usado em tradução automática
        # Desvantagem: Não captura semântica, pode penalizar sinônimos
        # ============================================================================
        if BLEU_AVAILABLE:
            bleu_scores = []
            for ref, pred in zip(references, predictions):
                try:
                    # Converte para minúsculas para comparação uniforme
                    ref_tokens = ref.lower().split()
                    pred_tokens = pred.lower().split()

                    # Calcula BLEU com pesos iguais para 1-4 gramas
                    # weights=(0.25, 0.25, 0.25, 0.25) significa:
                    # - 25% baseado em palavras individuais (1-gramas)
                    # - 25% baseado em pares de palavras (2-gramas)
                    # - 25% baseado em triplas de palavras (3-gramas)
                    # - 25% baseado em quadruplas de palavras (4-gramas)
                    smoothing = SmoothingFunction().method1
                    bleu = sentence_bleu(
                        [ref_tokens],  # Referência em lista (pode ter múltiplas)
                        pred_tokens,   # Predição a avaliar
                        weights=(0.25, 0.25, 0.25, 0.25),  # Pesos iguais para n-gramas
                        smoothing_function=smoothing  # Evita score 0
                    )
                    bleu_scores.append(bleu)
                except Exception as e:
                    logger.warning(f"Erro ao calcular BLEU: {e}")
                    bleu_scores.append(0.0)

            # Média de todos os BLEUs calculados
            metrics['bleu'] = np.mean(bleu_scores) if bleu_scores else 0.0
            metrics['bleu_scores_individual'] = bleu_scores
        else:
            metrics['bleu'] = None
            logger.warning("BLEU não disponível (instale rouge-score)")

        # ============================================================================
        # MÉTRICA 2B: METEOR (Metric for Evaluation of Translation with Explicit Ordering)
        # ============================================================================
        # O que é: Avalia qualidade de geração de texto considerando sinônimos e paráfrases
        #          Desenvolvido para tradução automática mas funciona bem para QA também
        #
        # Como funciona:
        # - Encontra matches de palavras entre predição e referência
        # - Tipos de match:
        #   1. EXACT: Palavra idêntica ("syndrome" = "syndrome")
        #   2. STEM: Mesma raiz ("syndromes" = "syndrome" por stemming)
        #   3. SYNONYM: Sinônimo ("illness" = "disease" por WordNet)
        # - Calcula Precision: matches / len(prediction)
        # - Calcula Recall: matches / len(reference)
        # - Aplica penalidade por falta de ordem ("fragile" vs "fragile X")
        # - Score de 0 a 1 (1 = match perfeito com ordem correta)
        #
        # Vantagem: Tolera sinônimos, variações morfológicas, diferentes ordens
        # Desvantagem: Depende de dicionários (WordNet), mais lento que BLEU
        #
        # Exemplo:
        # Esperado: "Fragile X chromosome syndrome"
        # Predição: "Fragile X Syndrome"
        # METEOR: Alto score (mesmo sendo mais curto, palavras principais matcham)
        # ============================================================================
        if METEOR_AVAILABLE:
            meteor_scores = []
            meteor_failed = False
            for ref, pred in zip(references, predictions):
                try:
                    # Tokeniza AMBOS em listas de tokens
                    ref_tokens = ref.lower().split()
                    pred_tokens = pred.lower().split()

                    # Calcula METEOR
                    meteor = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(meteor)
                except LookupError:
                    # WordNet não encontrado
                    logger.warning("WordNet não encontrado para METEOR. Desabilitando métrica.")
                    meteor_failed = True
                    break
                except Exception as e:
                    logger.warning(f"Erro ao calcular METEOR: {e}")
                    meteor_scores.append(0.0)

            if meteor_failed:
                metrics['meteor'] = None
            elif meteor_scores:
                metrics['meteor'] = np.mean(meteor_scores)
                metrics['meteor_scores_individual'] = meteor_scores
                logger.info(f"METEOR calculado: {metrics['meteor']:.4f}")
            else:
                metrics['meteor'] = None
        else:
            metrics['meteor'] = None

        # ============================================================================
        # MÉTRICA 2C: BERTScore
        # ============================================================================
        # O que é: Compara textos usando embeddings de BERT (Transformers)
        #          Captura semântica profunda e contexto científico/médico
        #
        # Como funciona:
        # - Converte cada token em embedding de alta dimensão usando BERT
        # - Compara similaridade de cosseno token-a-token
        # - Calcula Precision: média de matches mais similares em predição
        # - Calcula Recall: média de matches mais similares em referência
        # - F1-score: média harmônica entre Precision e Recall
        # - Score de 0 a 1 (1 = match semântico perfeito)
        #
        # Especialmente bom para:
        # - Textos científicos/médicos (BERT entende domínio)
        # - Paráfrases e sinônimos contextuais
        # - Respostas com ordem diferente mas significado igual
        #
        # Exemplo:
        # Esperado: "Fragile X chromosome syndrome"
        # Predição: "The disease caused by FMR1 gene mutation is Fragile X"
        # BERTScore: Reconhece que "FMR1" e "Fragile X" estão relacionados
        # (BLEU/ROUGE não conseguiriam)
        #
        # Vantagem: Captura semântica profunda, ideal para biomedicina
        # Desvantagem: Mais lento, requer GPU (opcional)
        # ============================================================================
        if BERTSCORE_AVAILABLE:
            try:
                # Usa modelo BERT padrão (english)
                # Pode usar modelo específico com lang="en"
                P, R, F1_scores = bert_score(predictions, references, lang="en", verbose=False)

                # P = Precision (quanto da predição está na referência)
                # R = Recall (quanto da referência está na predição)
                # F1 = média harmônica entre P e R

                metrics['bertscore_precision'] = P.mean().item()
                metrics['bertscore_recall'] = R.mean().item()
                metrics['bertscore_f1'] = F1_scores.mean().item()
                metrics['bertscore_f1_scores_individual'] = F1_scores.tolist()

                logger.info(f"BERTScore calculado - F1: {metrics['bertscore_f1']:.4f}")
            except Exception as e:
                logger.warning(f"Erro ao calcular BERTScore: {e}")
                metrics['bertscore_precision'] = None
                metrics['bertscore_recall'] = None
                metrics['bertscore_f1'] = None
        else:
            metrics['bertscore_precision'] = None
            metrics['bertscore_recall'] = None
            metrics['bertscore_f1'] = None
            logger.warning("BERTScore não disponível (instale com: pip install bert-score)")

        # ============================================================================
        # MÉTRICA 3: Similaridade Semântica (Embedding-based)
        # ============================================================================
        # O que é: Compara o SIGNIFICADO (semântica) entre textos usando embeddings
        #          em vez de apenas palavras/n-gramas
        #
        # Como funciona:
        # - Converte cada texto em um vetor numérico de alta dimensão (384 dimensões)
        #   que captura o significado semântico
        # - Calcula similaridade de cosseno entre os dois vetores
        # - Score de 0 a 1 (1 = significado idêntico)
        # - Dois textos com palavras diferentes mas mesmo significado têm score alto
        #
        # Exemplo:
        # - "O gato é preto" vs "Um felino negro" = alta similaridade (mesmo significado)
        # - "O gato é preto" vs "A mulher é alta" = baixa similaridade (significado diferente)
        #
        # Vantagem: Captura semântica real, tolera sinônimos e paráfrases
        # Desvantagem: Mais lento, requer mais poder computacional
        # ============================================================================
        if EMBEDDING_AVAILABLE:
            try:
                semantic_scores = self._calculate_semantic_similarity(references, predictions)
                metrics['semantic_similarity'] = np.mean(semantic_scores) if semantic_scores else 0.0
                metrics['semantic_similarity_scores_individual'] = semantic_scores
            except Exception as e:
                logger.warning(f"Erro ao calcular similaridade semântica: {e}")
                metrics['semantic_similarity'] = None
        else:
            metrics['semantic_similarity'] = None
            logger.warning("Similaridade semântica não disponível (instale sentence-transformers)")

        # ============================================================================
        # Compilar scores individuais de cada questão
        # ============================================================================
        # Para cada questão, cria um dicionário com todas as métricas
        # Permite análise por questão, não apenas média geral
        for i in range(len(references)):
            score_dict = {'index': i}

            # Adiciona BLEU se disponível
            if metrics.get('bleu') is not None and metrics.get('bleu_scores_individual'):
                score_dict['bleu'] = metrics['bleu_scores_individual'][i]

            # Adiciona ROUGE-L se disponível
            if metrics.get('rouge_l') is not None and metrics.get('rouge_l_scores_individual'):
                score_dict['rouge_l'] = metrics['rouge_l_scores_individual'][i]

            # Adiciona METEOR se disponível
            if metrics.get('meteor') is not None and metrics.get('meteor_scores_individual'):
                score_dict['meteor'] = metrics['meteor_scores_individual'][i]

            # Adiciona BERTScore se disponível
            if metrics.get('bertscore_f1') is not None and metrics.get('bertscore_f1_scores_individual'):
                score_dict['bertscore_f1'] = metrics['bertscore_f1_scores_individual'][i]

            # Adiciona Similaridade Semântica se disponível
            if metrics.get('semantic_similarity') is not None and metrics.get('semantic_similarity_scores_individual'):
                score_dict['semantic_similarity'] = metrics['semantic_similarity_scores_individual'][i]

            metrics['individual_scores'].append(score_dict)

        logger.info(f"✅ Métricas de geração calculadas")
        return metrics

    def _calculate_semantic_similarity(
        self,
        references: List[str],
        predictions: List[str]
    ) -> List[float]:
        """
        Calcula similaridade semântica usando embeddings.

        Args:
            references: Lista de textos de referência
            predictions: Lista de textos preditos

        Returns:
            Lista de scores (0-1)
        """
        try:
            if self.embedding_model is None:
                logger.info(f"Carregando modelo de embedding: {self._embedding_model_name}")
                self.embedding_model = SentenceTransformer(self._embedding_model_name)

            # Gera embeddings
            ref_embeddings = self.embedding_model.encode(references, convert_to_tensor=False)
            pred_embeddings = self.embedding_model.encode(predictions, convert_to_tensor=False)

            # Calcula similaridade de cosseno
            from sklearn.metrics.pairwise import cosine_similarity

            scores = []
            for ref_emb, pred_emb in zip(ref_embeddings, pred_embeddings):
                sim = cosine_similarity([ref_emb], [pred_emb])[0][0]
                scores.append(float(sim))

            return scores

        except Exception as e:
            logger.error(f"Erro ao calcular similaridade semântica: {e}")
            raise

    def format_metrics_report(
        self,
        dataset_type: str,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Formata métricas para exibição legível.

        Args:
            dataset_type: 'closed' ou 'open'
            metrics: Dict com métricas

        Returns:
            String formatada para exibição
        """
        report = ""

        if dataset_type == 'closed' or dataset_type == DatasetType.CLOSED.value:
            report += "📊 MÉTRICAS DE CLASSIFICAÇÃO\n"
            report += "=" * 60 + "\n"
            report += f"Acurácia:  {metrics.get('accuracy', 0):.2%}\n"
            report += f"Precisão:  {metrics.get('precision', 0):.2%}\n"
            report += f"Recall:    {metrics.get('recall', 0):.2%}\n"
            report += f"F1-Score:  {metrics.get('f1', 0):.2%}\n"

        else:  # open
            report += "📝 MÉTRICAS DE GERAÇÃO DE TEXTO\n"
            report += "=" * 60 + "\n"

            if metrics.get('bleu') is not None:
                report += f"BLEU:                  {metrics['bleu']:.4f}\n"
            if metrics.get('rouge_l') is not None:
                report += f"ROUGE-L:               {metrics['rouge_l']:.4f}\n"
            if metrics.get('semantic_similarity') is not None:
                report += f"Similaridade Semântica: {metrics['semantic_similarity']:.4f}\n"

            report += f"\nItems avaliados: {metrics.get('count', 0)}\n"

        return report
