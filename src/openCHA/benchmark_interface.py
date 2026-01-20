import gradio as gr
import logging
import time
import re
import json
from typing import Callable, Tuple, Optional, Dict, Any
from openCHA.dataset_tools import GenericDatasetLoader, MetricsSelector, DatasetDetector
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


class BenchmarkInterface:
    def __init__(self):
        self.dataset_loader = None
        self.evaluator = BenchmarkEvaluator()
        self.metrics_selector = MetricsSelector()
        self.detector = DatasetDetector()
        self.current_mapping = None
        self.current_dataset_type = None

    def extract_model_response(self, full_response: str, model_name: str) -> str:
        """
        Extrai a resposta de um modelo específico do relatório Multi-LLM.

        Args:
            full_response: Resposta completa formatada Multi-LLM
            model_name: Nome do modelo (chatgpt, gemini, deepseek)

        Returns:
            str: Resposta do modelo ou vazio se não encontrar
        """
        try:
            model_upper = model_name.upper()

            # Padrão: busca "🤖 MODELO\n" até a próxima seção com "=" ou fim
            pattern = rf"🤖\s+{model_upper}.*?📝\s+Resposta:(.*?)(?:🏆|={10,}|$)"
            match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)

            if match:
                resposta = match.group(1).strip()
                return resposta

            # Fallback: busca apenas a seção com "MODEL:"
            if f"{model_upper}" in full_response:
                sections = full_response.split(f"{model_upper}")
                if len(sections) > 1:
                    section = sections[1].split("=" * 80)[0]
                    lines = [l for l in section.split('\n') if not l.strip().startswith(('⏱️', '├', '└', '📝'))]
                    resposta = '\n'.join(lines).strip()
                    if resposta:
                        return resposta

            return ""

        except Exception as e:
            logger.error(f"Erro ao extrair resposta de {model_name}: {e}")
            return ""

    def process_json_upload(self, file) -> Tuple[str, str, str]:
        """
        Processa upload de arquivo JSON e tenta detectar estrutura.

        Args:
            file: Arquivo enviado pelo Gradio (pode ser bytes ou file-like object)

        Returns:
            Tuple: (mensagem_status, detecção_json, campo_pergunta)
        """
        try:
            # Lê arquivo - Gradio pode retornar bytes diretamente ou file-like object
            if isinstance(file, bytes):
                content = file.decode('utf-8')
            else:
                content = file.read().decode('utf-8')

            # Carrega com o GenericDatasetLoader
            self.dataset_loader = GenericDatasetLoader()
            normalized_data, mapping = self.dataset_loader.load_from_json(content)

            # Detecta tipo
            self.current_dataset_type = mapping['dataset_type']
            self.current_mapping = mapping

            # Gera mensagem de status
            detection = self.dataset_loader.get_detection_result()
            stats = self.dataset_loader.get_stats()

            status_msg = f"""
✅ **Dataset carregado com sucesso!**

📊 **Estatísticas:**
- Total de items: {stats['total_items']}
- Tipo detectado: {self.current_dataset_type.upper()}
- Valores únicos em resposta: {stats['unique_answers']}

🔍 **Detecção Automática:**
- Campo de pergunta: `{mapping['question_field']}` ({detection['question_confidence']:.0%})
- Campo de resposta: `{mapping['answer_field']}` ({detection['answer_confidence']:.0%})
- Campo de ID: `{mapping['id_field']}` (opcional)
- Confiança geral: {detection['overall_confidence']:.0%}

⚠️ {'Confirmação necessária (confiança baixa)' if mapping['needs_confirmation'] else '✅ Mapeamento aprovado automaticamente'}
            """

            detection_json = json.dumps({
                'question_field': mapping['question_field'],
                'answer_field': mapping['answer_field'],
                'id_field': mapping['id_field'],
                'dataset_type': self.current_dataset_type,
                'confidence': detection['overall_confidence']
            }, indent=2)

            suggestions = self.dataset_loader.get_mapping_suggestions()
            suggestions_msg = "\n".join([
                f"- Opção {s['rank']}: `{s['mapping']['question_field']}` → `{s['mapping']['answer_field']}`"
                for s in suggestions[:3]
            ])

            return status_msg, detection_json, suggestions_msg

        except Exception as e:
            error_msg = f"❌ Erro ao processar arquivo: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""

    def apply_mapping(self, mapping_json: str) -> str:
        """
        Aplica um mapeamento customizado.

        Args:
            mapping_json: JSON com mapeamento

        Returns:
            Mensagem de status
        """
        try:
            if self.dataset_loader is None or self.dataset_loader.raw_data is None:
                return "❌ Nenhum dataset carregado"

            mapping = json.loads(mapping_json)

            # Valida mapping
            validation = self.detector.validate_mapping(
                self.dataset_loader.raw_data,
                {
                    'question_field': mapping['question_field'],
                    'answer_field': mapping['answer_field'],
                    'id_field': mapping['id_field']
                }
            )

            if not validation['valid']:
                return f"❌ Mapeamento inválido:\n" + "\n".join(validation['errors'])

            # Aplica mapping customizado
            self.dataset_loader.apply_custom_mapping(
                data_list=self.dataset_loader.raw_data,
                mapping={
                    'question_field': mapping['question_field'],
                    'answer_field': mapping['answer_field'],
                    'id_field': mapping['id_field']
                }
            )

            self.current_mapping = mapping

            return f"""
✅ **Mapeamento customizado aplicado!**

- Pergunta: `{mapping['question_field']}`
- Resposta: `{mapping['answer_field']}`
- ID: `{mapping['id_field']}`

Agora você pode iniciar o benchmark com os 3 modelos.
            """

        except json.JSONDecodeError:
            return "❌ JSON de mapeamento inválido"
        except Exception as e:
            return f"❌ Erro ao aplicar mapeamento: {str(e)}"

    def prepare_benchmark_tab(self, run_single_question: Callable, reset_fn: Callable):
        """
        Prepara aba de benchmark com upload de dataset e avaliação.

        Args:
            run_single_question: Função do openCHA que roda pergunta com orquestração
            reset_fn: Função de reset
        """
        with gr.Column():
            gr.Markdown("# 📊 Benchmark Flexível - Qualquer Dataset JSON")

            # ==================== SEÇÃO 1: UPLOAD ====================
            with gr.Group():
                gr.Markdown("## 📁 Etapa 1: Upload do Dataset")

                file_upload = gr.File(
                    label="📤 Selecione arquivo JSON",
                    file_types=[".json"],
                    type="binary"
                )

                upload_btn = gr.Button("🔍 Processar JSON", variant="primary")

                # Outputs do upload
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=8
                )

                with gr.Row():
                    detection_output = gr.Code(
                        label="Detecção Automática (JSON)",
                        language="json",
                        lines=8
                    )
                    suggestions_output = gr.Textbox(
                        label="Sugestões de Mapeamento",
                        interactive=False,
                        lines=8
                    )

            # ==================== SEÇÃO 2: CONFIRMAÇÃO ====================
            with gr.Group():
                gr.Markdown("## ✏️ Etapa 2: Confirmar/Editar Mapeamento")

                mapping_input = gr.Code(
                    label="Edite o mapeamento se necessário (JSON)",
                    language="json",
                    value=json.dumps({
                        "question_field": "question",
                        "answer_field": "answer",
                        "id_field": None,
                        "dataset_type": "open"
                    }, indent=2),
                    lines=10
                )

                apply_mapping_btn = gr.Button("✅ Aplicar Mapeamento", variant="primary")
                mapping_status = gr.Textbox(
                    label="Status do Mapeamento",
                    interactive=False,
                    lines=4
                )

            # ==================== SEÇÃO 3: BENCHMARK ====================
            with gr.Group():
                gr.Markdown("## 🚀 Etapa 3: Executar Benchmark")

                with gr.Row():
                    models_to_test = gr.CheckboxGroup(
                        label="Modelos",
                        choices=["chatgpt", "gemini", "deepseek"],
                        value=["chatgpt", "gemini", "deepseek"]
                    )

                    num_samples = gr.Slider(
                        label="Número de questões",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=3
                    )

                btn_start = gr.Button("🚀 Iniciar Benchmark Paralelo", variant="primary")

                progress = gr.Textbox(
                    label="Progresso",
                    interactive=False,
                    lines=4
                )

                result_text = gr.Textbox(
                    label="Resultados Detalhados",
                    interactive=False,
                    lines=50
                )

            # ==================== CALLBACKS ====================

            def handle_upload(file):
                """Processa upload"""
                if file is None:
                    return "❌ Nenhum arquivo selecionado", "", ""

                status, detection, suggestions = self.process_json_upload(file)

                # Atualiza o input de mapeamento com a detecção
                if self.current_mapping:
                    mapping_json = json.dumps({
                        "question_field": self.current_mapping['question_field'],
                        "answer_field": self.current_mapping['answer_field'],
                        "id_field": self.current_mapping['id_field'],
                        "dataset_type": self.current_mapping['dataset_type']
                    }, indent=2)
                else:
                    mapping_json = json.dumps({
                        "question_field": "question",
                        "answer_field": "answer",
                        "id_field": None
                    }, indent=2)

                return status, detection, suggestions

            def handle_apply_mapping(mapping_str):
                """Aplica mapeamento customizado"""
                return self.apply_mapping(mapping_str)

            def run_benchmark(models, num_samples):
                """Executa benchmark"""
                if self.dataset_loader is None or self.dataset_loader.data is None:
                    return "❌ Nenhum dataset carregado. Faça upload primeiro.", ""

                results_text = f"📖 Benchmark iniciado com {len(self.dataset_loader.data)} questões\n"
                results_text += f"📊 Tipo: {self.current_dataset_type.upper()}\n"
                results_text += f"🤖 Modelos: {', '.join(models)}\n"
                results_text += f"{'='*80}\n\n"

                try:
                    # Pega subset
                    questions = self.dataset_loader.get_subset(num_samples)

                    # Inicializa storage de resultados
                    all_results = {
                        model: {"correct": 0, "total": 0, "scores": []}
                        for model in models
                    }

                    # Processa cada questão
                    for i, q in enumerate(questions, 1):
                        results_text += f"{'='*80}\n"
                        results_text += f"❓ QUESTÃO {i}/{len(questions)}\n"
                        results_text += f"{'='*80}\n"
                        results_text += f"Pergunta: {q['question'][:200]}...\n"
                        results_text += f"Resposta esperada: {q['expected_answer'][:100]}...\n"
                        results_text += f"{'-'*80}\n"

                        try:
                            start = time.time()

                            # Executa Multi-LLM paralelo
                            full_response = run_single_question(
                                q['question'],
                                use_multi_llm=True,
                                compare_models=models
                            )

                            time_ms = (time.time() - start) * 1000

                            # Processa resposta de cada modelo
                            for model in models:
                                try:
                                    model_response = self.extract_model_response(full_response, model)

                                    # Avalia baseado no tipo de dataset
                                    if self.current_dataset_type == 'closed':
                                        eval_result = self.evaluator.evaluate(
                                            q['expected_answer'],
                                            model_response
                                        )
                                        icon = "✅" if eval_result["correct"] else "❌"
                                        results_text += f"\n{icon} {model.upper()}\n"
                                        results_text += f"   Resposta: {model_response[:150]}\n"
                                        results_text += f"   Detectado: {eval_result['extracted']} ({time_ms/len(models):.0f}ms)\n"

                                        all_results[model]["total"] += 1
                                        if eval_result["correct"]:
                                            all_results[model]["correct"] += 1

                                    else:  # open
                                        results_text += f"\n🔄 {model.upper()}\n"
                                        results_text += f"   Resposta: {model_response[:150]}...\n"
                                        results_text += f"   ({time_ms/len(models):.0f}ms)\n"

                                        all_results[model]["total"] += 1
                                        # Armazena resposta para cálculo de métricas depois
                                        if "responses" not in all_results[model]:
                                            all_results[model]["responses"] = []
                                        all_results[model]["responses"].append(model_response)

                                except Exception as e:
                                    results_text += f"\n❌ {model.upper()}\n"
                                    results_text += f"   Erro: {str(e)}\n"
                                    logger.error(f"Erro ao processar {model}: {e}")

                        except Exception as e:
                            results_text += f"\n❌ Erro na questão {i}: {str(e)}\n"
                            logger.error(f"Erro na questão {i}: {e}")

                    # Resumo final
                    results_text += f"\n{'='*80}\n"
                    results_text += "🏆 RESUMO FINAL:\n"
                    results_text += f"{'='*80}\n"

                    if self.current_dataset_type == 'closed':
                        for model in models:
                            total = all_results[model]["total"]
                            if total > 0:
                                acc = all_results[model]["correct"] / total
                                results_text += f"{model.upper()}: {acc:.0%} ({all_results[model]['correct']}/{total})\n"
                            else:
                                results_text += f"{model.upper()}: Nenhuma questão processada\n"
                    else:
                        # Para dataset aberto, calcular métricas de similaridade
                        results_text += "\n📊 MÉTRICAS DE SIMILARIDADE:\n"
                        results_text += f"{'-'*80}\n"

                        # Coleta todas as respostas esperadas
                        expected_answers = [q['expected_answer'] for q in questions]

                        for model in models:
                            try:
                                if "responses" in all_results[model] and len(all_results[model]["responses"]) > 0:
                                    # Calcula métricas
                                    metrics = self.metrics_selector.calculate_open_metrics(
                                        expected_answers,
                                        all_results[model]["responses"]
                                    )

                                    # Exibe resultados
                                    results_text += f"\n🤖 {model.upper()}:\n"

                                    if metrics.get('bleu') is not None:
                                        results_text += f"   BLEU:                    {metrics['bleu']:.4f}\n"
                                    if metrics.get('rouge_l') is not None:
                                        results_text += f"   ROUGE-L:                 {metrics['rouge_l']:.4f}\n"
                                    if metrics.get('meteor') is not None:
                                        results_text += f"   METEOR:                  {metrics['meteor']:.4f} ⭐\n"
                                    if metrics.get('bertscore_f1') is not None:
                                        results_text += f"   BERTScore F1:            {metrics['bertscore_f1']:.4f} ⭐⭐\n"
                                    if metrics.get('semantic_similarity') is not None:
                                        results_text += f"   Similaridade Semântica:  {metrics['semantic_similarity']:.4f}\n"

                            except Exception as e:
                                results_text += f"\n❌ Erro ao calcular métricas para {model}: {str(e)}\n"
                                logger.error(f"Erro ao calcular métricas para {model}: {e}")

                    return results_text, results_text

                except Exception as e:
                    error_msg = f"❌ Erro ao executar benchmark: {str(e)}"
                    logger.error(error_msg)
                    return error_msg, error_msg

            # Conecta callbacks
            upload_btn.click(
                fn=handle_upload,
                inputs=[file_upload],
                outputs=[upload_status, detection_output, suggestions_output]
            )

            apply_mapping_btn.click(
                fn=handle_apply_mapping,
                inputs=[mapping_input],
                outputs=[mapping_status]
            )

            btn_start.click(
                fn=run_benchmark,
                inputs=[models_to_test, num_samples],
                outputs=[progress, result_text]
            )
