#!/usr/bin/env python3
"""
openCHA + Multi-LLM - Teste Simples
Compare ChatGPT, Gemini e DeepSeek lado a lado
"""
from openCHA import openCHA


def main():
    print("🔷 openCHA + Multi-LLM")
    print("=" * 50)
    
    # Cria o agente com Multi-LLM habilitado
    cha = openCHA(
        name="openCHA-MultiLLM",
        verbose=False,
        
        # Configurações Multi-LLM
        multi_llm_enable_cache=True,
        multi_llm_timeout=180,
        multi_llm_max_workers=3,
    )
    
    print("🌐 Iniciando interface web...")
    print("📍 URL: http://localhost:7860")
    print("🤖 Modelos: ChatGPT | Gemini | DeepSeek")
    print("✨ Modo: Comparação Multi-LLM")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 50)
    print()
    print("💡 Como usar:")
    print("  1. Configure suas API keys na interface")
    print("  2. Ative 'Modo Multi-LLM' no accordion")
    print("  3. Selecione os modelos para comparar")
    print("  4. Digite sua pergunta!")
    print()
    
    try:
        cha.run_with_interface()
    except KeyboardInterrupt:
        print("\n👋 openCHA encerrado")


if __name__ == "__main__":
    main()
