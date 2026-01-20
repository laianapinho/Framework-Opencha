#!/usr/bin/env python3
"""
openCHA + Benchmark Flexível - Com Orquestração
Teste 3 modelos com qualquer dataset JSON (PubMedQA, CareQA, customizado, etc)
"""
import sys
sys.path.insert(0, '/home/laiana/Framework-Opencha/src')
import os
from openCHA.openCHA import openCHA
from openCHA.benchmark_interface import BenchmarkInterface


def main():
    print("📊 openCHA + Benchmark Flexível (Qualquer Dataset JSON)")
    print("=" * 70)

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
    print("✨ Modo: Benchmark Flexível (upload qualquer JSON)")
    print("📁 Suporta: PubMedQA, CareQA, e qualquer estrutura JSON customizada")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 70)
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

        with gr.Blocks(theme=gr.themes.Soft(), title="openCHA - Benchmark Flexível") as demo:
            gr.Markdown("# 🔷 openCHA + Benchmark Flexível")
            gr.Markdown("### 📊 Avalie qualquer dataset JSON com 3 modelos em paralelo")

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

                # ABA 2: Benchmark Flexível
                with gr.Tab("📊 Benchmark Flexível"):
                    gr.Markdown("""
                    ### 🚀 Como usar:
                    1. **Upload**: Selecione um arquivo JSON com suas perguntas e respostas
                    2. **Detecção**: O sistema detecta automaticamente a estrutura do JSON
                    3. **Confirmação**: Confirme ou edite o mapeamento de campos
                    4. **Benchmark**: Execute a avaliação com os 3 modelos

                    ### ✅ Formatos suportados:
                    - **PubMedQA**: `{"QUESTION": "...", "final_decision": "yes/no/maybe"}`
                    - **CareQA**: `[{"question": "...", "answer": "..."}, ...]`
                    - **Customizado**: Qualquer JSON com pergunta e resposta esperada
                    """)

                    benchmark = BenchmarkInterface()
                    benchmark.prepare_benchmark_tab(
                        run_single_question=cha.run,
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
