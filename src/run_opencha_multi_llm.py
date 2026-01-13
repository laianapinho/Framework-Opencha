#!/usr/bin/env python3
"""
openCHA + Multi-LLM - Teste Simples (CORRIGIDO)
Compare ChatGPT, Gemini e DeepSeek lado a lado

✅ CORREÇÃO: Pré-inicializa os modelos ANTES de abrir a interface
   Evita race condition e APIConnectionError
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

    print("🌐 Pré-inicializando modelos...")
    print("-" * 50)

    # ✅ CORREÇÃO: Força inicialização ANTES de abrir a interface
    # Isso evita race condition quando a interface tenta usar os modelos
    try:
        manager = cha.get_multi_llm()
        modelos_disponiveis = manager.get_available_models()
        print(f"✅ Modelos prontos: {', '.join(modelos_disponiveis)}")
        print(f"✅ Total: {len(modelos_disponiveis)} modelo(s) inicializado(s)")
    except Exception as e:
        print(f"⚠️ Erro na pré-inicialização: {e}")
        print("   Tentando continuar mesmo assim...")

    print("-" * 50)
    print()

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
