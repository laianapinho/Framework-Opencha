"""
Interface Médica Simplificada para openCHA + Benchmark
========================================================

Design focado em MÉDICOS:
✅ Linguagem clara e simples
✅ Sem jargão técnico
✅ 3 etapas visuais
✅ Resultados em formato claro
✅ Design profissional e limpo
"""

import gradio as gr
import logging
import time
import re
import json
from typing import Callable, Tuple, Optional, Dict, Any
from openCHA.dataset_tools import GenericDatasetLoader, MetricsSelector, DatasetDetector
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


class MedicalBenchmarkInterface:
    """Interface simplificada para médicos avaliarem modelos de IA"""

    def __init__(self):
        self.dataset_loader = None
        self.evaluator = BenchmarkEvaluator()
        self.metrics_selector = MetricsSelector()
        self.detector = DatasetDetector()
        self.current_mapping = None
        self.current_dataset_type = None

    def extract_model_response(self, full_response: str, model_name: str) -> str:
        """Extrai resposta de um modelo específico"""
        try:
            model_upper = model_name.upper()
            pattern = rf"🤖\s+{model_upper}.*?📝\s+Resposta:(.*?)(?:🏆|={10,}|$)"
            match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)

            if match:
                resposta = match.group(1).strip()
                return resposta

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

    def process_json_upload(self, file) -> Tuple[str, str]:
        """
        Processa upload de arquivo JSON de forma silenciosa.
        Retorna apenas status amigável para médicos.
        """
        try:
            if isinstance(file, bytes):
                content = file.decode('utf-8')
            else:
                content = file.read().decode('utf-8')

            self.dataset_loader = GenericDatasetLoader()
            normalized_data, mapping = self.dataset_loader.load_from_json(content)
            self.current_dataset_type = mapping['dataset_type']
            self.current_mapping = mapping

            stats = self.dataset_loader.get_stats()

            status_msg = f"""
✅ **Arquivo carregado com sucesso!**

📊 **Informações do dataset:**
• Total de questões: {stats['total_items']}
• Tipo: {self.current_dataset_type.upper()}

Você pode prosseguir para a próxima etapa.
            """

            return status_msg, "ok"

        except Exception as e:
            error_msg = f"❌ Erro ao processar arquivo: {str(e)}\n\nTente com outro arquivo JSON."
            logger.error(error_msg)
            return error_msg, "error"

    def create_interface(self, run_single_question: Callable, reset_fn: Callable) -> gr.Blocks:
        """Cria a interface Gradio simplificada"""

        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="openCHA - Avaliação de Modelos",
            css="""
            .medical-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .step-container {
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 15px 0;
                background: #f8f9fa;
                border-radius: 5px;
            }
            .metric-box {
                background: white;
                border: 1px solid #e0e0e0;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            """
        ) as demo:

            # ============== HEADER ==============
            gr.HTML("""
            <div class="medical-header">
                <h1>🔷 openCHA - Avaliação de Modelos de IA</h1>
                <p style="font-size: 16px; margin-top: 10px;">
                    Sistema simples para avaliar como ChatGPT, Gemini e DeepSeek
                    respondem a perguntas sobre saúde e medicina
                </p>
            </div>
            """)

            # ============== ETAPA 1: UPLOAD ==============
            with gr.Group():
                gr.Markdown("## 📁 Etapa 1: Carregue seu arquivo de perguntas")
                gr.Markdown(
                    "Selecione um arquivo JSON com suas questões de saúde/medicina "
                    "e as respostas esperadas. O sistema detectará automaticamente o formato."
                )

                with gr.Row():
                    file_upload = gr.File(
                        label="Selecione arquivo JSON",
                        file_types=[".json"],
                        type="binary"
                    )

                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=4,
                    show_label=True
                )

                upload_btn = gr.Button(
                    "✅ Carregar Arquivo",
                    variant="primary",
                    size="lg"
                )

            # ============== ETAPA 2: SELECIONAR MODELOS ==============
            with gr.Group():
                gr.Markdown("## 🤖 Etapa 2: Escolha os modelos para comparação")
                gr.Markdown(
                    "Selecione quais modelos você deseja avaliar. "
                    "Todos requerem chaves de API configuradas."
                )

                models_to_test = gr.CheckboxGroup(
                    label="Modelos disponíveis",
                    choices=[
                        ("ChatGPT (OpenAI)", "chatgpt"),
                        ("Gemini (Google)", "gemini"),
                        ("DeepSeek (Alibaba)", "deepseek")
                    ],
                    value=["chatgpt", "gemini", "deepseek"],
                    interactive=True
                )

                with gr.Row():
                    num_samples = gr.Slider(
                        label="Quantas questões avaliar?",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=3,
                        info="Escolha 1-50. Mais questões = mais tempo de processamento"
                    )

            # ============== ETAPA 3: EXECUTAR E VER RESULTADOS ==============
            with gr.Group():
                gr.Markdown("## 📊 Etapa 3: Execute a avaliação")

                btn_start = gr.Button(
                    "🚀 Iniciar Avaliação",
                    variant="primary",
                    size="lg"
                )

                # Progress
                progress_status = gr.Textbox(
                    label="Progresso",
                    interactive=False,
                    lines=3,
                    show_label=True
                )

                # Resultados
                result_text = gr.Textbox(
                    label="📈 Resultados da Avaliação",
                    interactive=False,
                    lines=30,
                    show_label=True
                )

            # ============== CALLBACKS ==============

            def handle_upload(file):
                """Processa upload"""
                if file is None:
                    return "❌ Selecione um arquivo antes de continuar"

                status, result = self.process_json_upload(file)
                return status

            def run_benchmark(models, num_samples):
                """Executa benchmark com interface simples"""
                if self.dataset_loader is None or self.dataset_loader.data is None:
                    return "❌ Nenhum arquivo carregado. Faça o upload primeiro.", ""

                results_text = ""
                progress_msg = ""

                try:
                    questions = self.dataset_loader.get_subset(num_samples)

                    # Storage de resultados
                    all_results = {
                        model: {"correct": 0, "total": 0, "scores": []}
                        for model in models
                    }

                    total_questions = len(questions)
                    progress_msg = f"Iniciando avaliação de {total_questions} questões com {len(models)} modelo(s)...\n"
                    progress_msg += f"Tempo estimado: ~{total_questions * len(models) * 5} segundos\n"

                    results_text += f"{'='*80}\n"
                    results_text += f"📊 AVALIAÇÃO DE MODELOS - {total_questions} QUESTÕES\n"
                    results_text += f"{'='*80}\n\n"

                    # Processa cada questão
                    for i, q in enumerate(questions, 1):
                        progress_msg = f"Processando questão {i}/{total_questions}..."

                        results_text += f"{'─'*80}\n"
                        results_text += f"❓ Questão {i}\n"
                        results_text += f"{'─'*80}\n"
                        results_text += f"Pergunta: {q['question']}\n"
                        results_text += f"Resposta esperada: {q['expected_answer']}\n\n"

                        try:
                            start = time.time()

                            # Executa Multi-LLM
                            full_response = run_single_question(
                                q['question'],
                                use_multi_llm=True,
                                compare_models=models
                            )

                            time_ms = (time.time() - start) * 1000

                            # Avalia cada modelo
                            for model in models:
                                try:
                                    model_response = self.extract_model_response(full_response, model)

                                    if self.current_dataset_type == 'closed':
                                        eval_result = self.evaluator.evaluate(
                                            q['expected_answer'],
                                            model_response
                                        )
                                        icon = "✅" if eval_result["correct"] else "❌"
                                        results_text += f"{icon} {model.upper()}: {eval_result['extracted']}\n"

                                        all_results[model]["total"] += 1
                                        if eval_result["correct"]:
                                            all_results[model]["correct"] += 1

                                    else:  # open
                                        results_text += f"📝 {model.upper()}:\n"
                                        results_text += f"    {model_response[:200]}...\n"
                                        all_results[model]["total"] += 1
                                        if "responses" not in all_results[model]:
                                            all_results[model]["responses"] = []
                                        all_results[model]["responses"].append(model_response)

                                except Exception as e:
                                    logger.error(f"Erro ao processar {model}: {e}")

                        except Exception as e:
                            results_text += f"❌ Erro na questão: {str(e)}\n"
                            logger.error(f"Erro: {e}")

                    # ============== RESUMO FINAL ==============
                    results_text += f"\n{'='*80}\n"
                    results_text += "🏆 RESUMO FINAL\n"
                    results_text += f"{'='*80}\n\n"

                    if self.current_dataset_type == 'closed':
                        results_text += "📊 ACURÁCIA POR MODELO:\n"
                        results_text += f"{'─'*80}\n"
                        for model in models:
                            total = all_results[model]["total"]
                            if total > 0:
                                acc = all_results[model]["correct"] / total * 100
                                bar_length = int(acc / 5)
                                bar = "█" * bar_length + "░" * (20 - bar_length)
                                results_text += f"{model.upper():10} {bar} {acc:6.1f}% ({all_results[model]['correct']}/{total})\n"
                            else:
                                results_text += f"{model.upper():10} Não processado\n"
                    else:
                        results_text += "📊 MÉTRICAS DE QUALIDADE:\n"
                        results_text += f"{'─'*80}\n"

                        expected_answers = [q['expected_answer'] for q in questions]

                        for model in models:
                            try:
                                if "responses" in all_results[model] and len(all_results[model]["responses"]) > 0:
                                    metrics = self.metrics_selector.calculate_open_metrics(
                                        expected_answers,
                                        all_results[model]["responses"]
                                    )

                                    results_text += f"\n{model.upper()}:\n"
                                    if metrics.get('bertscore_f1') is not None:
                                        results_text += f"  BERTScore F1: {metrics['bertscore_f1']:.4f} ⭐⭐\n"
                                    if metrics.get('meteor') is not None:
                                        results_text += f"  METEOR:       {metrics['meteor']:.4f} ⭐\n"
                                    if metrics.get('semantic_similarity') is not None:
                                        results_text += f"  Similaridade: {metrics['semantic_similarity']:.4f}\n"

                            except Exception as e:
                                logger.error(f"Erro ao calcular métricas: {e}")

                    results_text += f"\n{'='*80}\n"
                    results_text += "✅ Avaliação concluída!\n"

                    return results_text, results_text

                except Exception as e:
                    error_msg = f"❌ Erro durante avaliação: {str(e)}"
                    logger.error(error_msg)
                    return error_msg, error_msg

            # ============== CONECTAR CALLBACKS ==============
            upload_btn.click(
                fn=handle_upload,
                inputs=[file_upload],
                outputs=[upload_status]
            )

            btn_start.click(
                fn=run_benchmark,
                inputs=[models_to_test, num_samples],
                outputs=[progress_status, result_text]
            )

        return demo
