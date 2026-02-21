#!/usr/bin/env python3
"""
openCHA + Avaliação Médica de Modelos
======================================

Interface simplificada para médicos avaliarem modelos de IA
Foco: Clareza, Simplicidade, Sem Jargão Técnico
"""
import sys
sys.path.insert(0, '/home/laiana/Framework-Opencha/src')

import os
from openCHA.openCHA import openCHA
from openCHA.benchmark_interface_medical import MedicalBenchmarkInterface


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  🔷 openCHA - Avaliação de Modelos de IA para Saúde".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print()

    # Inicializa o agente
    cha = openCHA(
        name="openCHA-Medical-Benchmark",
        verbose=False,
        multi_llm_enable_cache=True,
        multi_llm_timeout=180,
        multi_llm_max_workers=3,
    )

    print("✨ Inicializando interface...")
    print()

    try:
        import gradio as gr

        # Cria interface médica simplificada
        medical_interface = MedicalBenchmarkInterface()
        interface = medical_interface.create_interface(
            run_single_question=cha.run,
            reset_fn=cha.reset
        )

        print("✅ Interface carregada com sucesso!")
        print()
        print("┌" + "─"*78 + "┐")
        print("│ 🌐 Abrindo aplicação no navegador...".ljust(79) + "│")
        print("│ 📍 URL: http://localhost:7860".ljust(79) + "│")
        print("│ 🤖 Modelos: ChatGPT | Gemini | DeepSeek".ljust(79) + "│")
        print("│ 📊 Modo: Avaliação Comparativa".ljust(79) + "│")
        print("│ 🛑 Para parar: Pressione Ctrl+C".ljust(79) + "│")
        print("└" + "─"*78 + "┘")
        print()

        interface.launch(
            share=False,
            server_port=7860,
            show_error=True,
            quiet=False
        )

    except KeyboardInterrupt:
        print("\n\n👋 Aplicação encerrada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
