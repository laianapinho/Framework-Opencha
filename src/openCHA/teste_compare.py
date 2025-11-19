from openCHA import openCHA

if __name__ == "__main__":
    cha = openCHA()

    # (Opcional) Definir as chaves aqui se não quiser usar interface
    import os
    os.environ["OPENAI_API_KEY"] = "chave api"
    os.environ["GEMINI_API_KEY"] = "chave api"
    os.environ["DEEPSEEK_API_KEY"] = "chave api"

    query = "Quais são os sintomas iniciais de pneumonia?"

    resultado = cha.run(
        query=query,
        use_multi_llm=True,                 # <<< isso ativa o comparador
        compare_models=["chatgpt", "gemini", "deepseek"],
        max_tokens=400,
        temperature=0.7,
    )

    print(resultado)
