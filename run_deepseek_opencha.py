 #!/usr/bin/env python3
"""
openCHA + DeepSeek - Interface Web
"""
from openCHA import openCHA

def main():
    print("🔷 openCHA + DeepSeek")
    print("=" * 50)

    cha = openCHA(
        name="openCHA-DeepSeek",
        planner_llm="deepseek",
        response_generator_llm="deepseek",
        verbose=True
    )

    print("🌐 Iniciando interface web...")
    print("📍 URL: http://localhost:7860")
    print("🔷 Modelo: DeepSeek-Chat")
    print("🛑 Para parar: Ctrl+C")
    print("=" * 50)

    try:
        cha.run_with_interface()
    except KeyboardInterrupt:
        print("\n👋 openCHA encerrado")

if __name__ == "__main__":
    main()
