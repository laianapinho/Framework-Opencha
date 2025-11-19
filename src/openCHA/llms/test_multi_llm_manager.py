from openCHA.llms.multi_llm_manager import MultiLLMManager

def main():
    manager = MultiLLMManager()

    query = "Quais são os sintomas iniciais de pneumonia e quando devo procurar atendimento médico?"
    result = manager.generate_all(query)

    print("\n=== RESPOSTAS ===")
    for model, resp in result["responses"].items():
        print(f"\n--- {model.upper()} ---")
        print(resp)

    print("\n=== TEMPOS ===")
    print(result["times"])

    print("\n=== ERROS ===")
    print(result["errors"])

if __name__ == "__main__":
    main()
