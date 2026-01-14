import gradio as gr
import logging
import time
import re
from typing import Callable
from openCHA.pubmedqa_loader import PubMedQALoader
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)

class BenchmarkInterface:
    def __init__(self):
        self.loader = PubMedQALoader()
        self.evaluator = BenchmarkEvaluator()

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
            # Busca a seção do modelo (ex: "CHATGPT")
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
                    # Pega texto até a próxima linha com "="
                    section = sections[1].split("=" * 80)[0]
                    # Remove linhas de timing
                    lines = [l for l in section.split('\n') if not l.strip().startswith(('⏱️', '├', '└', '📝'))]
                    resposta = '\n'.join(lines).strip()
                    if resposta:
                        return resposta

            return ""

        except Exception as e:
            logger.error(f"Erro ao extrair resposta de {model_name}: {e}")
            return ""

    def prepare_benchmark_tab(self, run_single_question: Callable, reset_fn: Callable):
        """
        Args:
            run_single_question: Função do openCHA que roda pergunta com orquestração
                                Pode rodar em modo Multi-LLM para paralelo
            reset_fn: Função de reset
        """
        with gr.Column():
            gr.Markdown("# 📊 Benchmark PubMedQA - 3 Questões (Paralelo)")

            models_to_test = gr.CheckboxGroup(
                label="Modelos",
                choices=["chatgpt", "gemini", "deepseek"],
                value=["chatgpt", "gemini", "deepseek"]
            )

            btn_start = gr.Button("🚀 Iniciar Benchmark Paralelo", variant="primary")
            progress = gr.Textbox(label="Progresso", interactive=False, lines=3)
            result_text = gr.Textbox(label="Resultados", interactive=False, lines=40)

            def run_benchmark(models):
                results_text = "📖 Carregando questões...\n"
                questions = self.loader.get_subset(3)

                all_results = {model: {"correct": 0, "total": 0} for model in models}

                for i, q in enumerate(questions, 1):
                    results_text += f"\n{'='*80}\n"
                    results_text += f"❓ QUESTÃO {i}\n"
                    results_text += f"{'='*80}\n"
                    results_text += f"Pergunta: {q['question']}\n"
                    results_text += f"Resposta esperada: {q['expected_answer'].upper()}\n"
                    results_text += f"{'-'*80}\n"

                    try:
                        # ✅ Chama Multi-LLM PARALELO
                        start = time.time()

                        full_response = run_single_question(
                            q['question'],
                            use_multi_llm=True,
                            compare_models=models
                        )

                        time_ms = (time.time() - start) * 1000

                        # Para cada modelo, extrai e avalia resposta
                        for model in models:
                            try:
                                # ✅ EXTRAI resposta do modelo
                                model_response = self.extract_model_response(full_response, model)

                                # ✅ AVALIA resposta
                                eval_result = self.evaluator.evaluate(
                                    q['expected_answer'],
                                    model_response
                                )

                                icon = "✅" if eval_result["correct"] else "❌"

                                # ✅ MOSTRA: modelo + resposta extraída + resultado
                                results_text += f"\n{icon} {model.upper()}\n"
                                results_text += f"   Resposta: {model_response[:200]}\n"
                                results_text += f"   Detectado: {eval_result['extracted']} ({time_ms/len(models):.0f}ms)\n"

                                all_results[model]["total"] += 1
                                if eval_result["correct"]:
                                    all_results[model]["correct"] += 1

                            except Exception as e:
                                results_text += f"\n❌ {model.upper()}\n"
                                results_text += f"   Erro: {str(e)}\n"
                                logger.error(f"Erro ao processar {model}: {e}")

                    except Exception as e:
                        results_text += f"\n❌ Erro na questão {i}: {str(e)}\n"
                        logger.error(f"Erro na questão {i}: {e}")

                # ✅ RESUMO FINAL
                results_text += f"\n{'='*80}\n"
                results_text += "🏆 RESUMO FINAL (PARALELO):\n"
                results_text += f"{'='*80}\n"

                for model in models:
                    total = all_results[model]["total"]
                    if total > 0:
                        acc = all_results[model]["correct"] / total
                        results_text += f"{model.upper()}: {acc:.0%} ({all_results[model]['correct']}/{total})\n"
                    else:
                        results_text += f"{model.upper()}: Nenhuma questão processada\n"

                return results_text, results_text

            btn_start.click(
                fn=run_benchmark,
                inputs=[models_to_test],
                outputs=[progress, result_text]
            )
