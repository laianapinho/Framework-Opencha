#!/usr/bin/env python3
"""
openCHA + Benchmark PubMedQA - Com Orquestração
Teste 3 modelos com questões médicas reais
"""
import sys
sys.path.insert(0, '/home/laiana/Framework-Opencha/src')

import os
from openCHA.openCHA import openCHA
from openCHA.benchmark_interface import BenchmarkInterface
def main():
    print("📊 openCHA + Benchmark PubMedQA")
    print("=" * 50)

    # Cria o agente com orquestração
    cha = openCHA(
        name="openCHA-Benchmark",
        verbose=False,
        multi_llm_enable_cache=True,
        multi_llm_timeout=180,
        multi_llm_max_workers=3,
    )

    print("🌐 Iniciando interface web...")
    print("📍 URL: http://localhost:7860")
    print("🤖 Modelos: ChatGPT | Gemini | DeepSeek")
    print("✨ Modo: Benchmark PubMedQA (3 questões com orquestração)")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 50)
    print()

    try:
        from openCHA.interface import Interface
        from openCHA.tasks import TASK_TO_CLASS
        import gradio as gr

        interface = Interface()

        respond = cha.respond
        reset = cha.reset
        upload_meta = cha.upload_meta
        available_tasks = [key.value for key in TASK_TO_CLASS.keys()]

        with gr.Blocks(theme=gr.themes.Soft(), title="openCHA") as demo:
            gr.Markdown("# 🔷 openCHA + Benchmark PubMedQA")

            with gr.Accordion("🔑 API Keys", open=True):
                with gr.Row():
                    openai_key = gr.Textbox(label="OpenAI", type="password")
                    gemini_key = gr.Textbox(label="Gemini", type="password")
                    deepseek_key = gr.Textbox(label="DeepSeek", type="password")
                    serp_key = gr.Textbox(label="SERP", type="password")

            with gr.Tabs():
                # ABA 1: Chat Normal
                with gr.Tab("💬 Chat"):
                    msg = gr.Textbox(placeholder="Digite mensagem...")
                    btn = gr.Button("Enviar")
                    output = gr.Textbox(interactive=False, lines=10)

                # ABA 2: Benchmark
                with gr.Tab("📊 Benchmark"):
                    benchmark = BenchmarkInterface()
                    benchmark.prepare_benchmark_tab(
                        run_single_question=cha.run,  # ✅ CORRIGIDO: usar cha.run
                        reset_fn=reset
                    )

        demo.launch(share=False, server_port=7860)

    except KeyboardInterrupt:
        print("\n👋 Benchmark encerrado")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
