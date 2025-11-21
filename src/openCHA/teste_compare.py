"""
Exemplo de uso do openCHA REFATORADO com MultiLLMManager com ORQUESTRAÇÃO COMPLETA

Diferenças principais:
- ANTES: Cada LLM recebia a query direto (sem planejamento)
- AGORA: Cada LLM tem seu próprio Orchestrator (pensa + escreve)

Resultado:
- Respostas mais profundas
- Comparação mais justa
- Execução mais lenta (mas mais completa)
"""

from openCHA import openCHA
from openCHA.llms import LLMType
import os


def exemplo_1_modo_normal():
    """
    MODO NORMAL: Usa um único LLM (rápido)
    """
    print("\n" + "="*80)
    print("EXEMPLO 1: MODO NORMAL (Um LLM)")
    print("="*80)

    # Criar openCHA com DeepSeek como planejador e gerador
    cha = openCHA(
        verbose=True,
        planner_llm=LLMType.DEEPSEEK,
        response_generator_llm=LLMType.DEEPSEEK
    )

    # Fazer uma pergunta
    resposta = cha.run("O que é câncer de mama?")
    print(resposta)


def exemplo_2_comparacao_completa():
    """
    MODO COMPARAÇÃO: Cada LLM pensa e escreve (novo!)
    """
    print("\n" + "="*80)
    print("EXEMPLO 2: MODO COMPARAÇÃO COM ORQUESTRAÇÃO COMPLETA")
    print("="*80)

    cha = openCHA()

    # Comparar 3 LLMs - CADA UM COM SEU PRÓPRIO ORCHESTRATOR
    resultado = cha.run(
        query="Explique o que são redes convulucionais?",
        use_multi_llm=True,  # ← Ativa comparação
        compare_models=['chatgpt', 'gemini', 'deepseek'],
        max_tokens=300,
        temperature=0.7
    )

    print(resultado)


def exemplo_3_analise_detalhada():
    """
    Análise detalhada com tempos de planejamento e geração
    """
    print("\n" + "="*80)
    print("EXEMPLO 3: ANÁLISE DETALHADA")
    print("="*80)

    cha = openCHA()

    # Obter comparação detalhada
    analise = cha.compare_and_analyze_full(
        query="Qual é a diferença entre IA e Machine Learning?",
        models=['chatgpt', 'gemini']
    )

    # Acessar dados estruturados
    print("\n--- RESPOSTAS ---")
    for modelo, resposta in analise['responses'].items():
        print(f"\n{modelo.upper()}:")
        print(resposta[:200] + "...")  # Primeiros 200 caracteres

    print("\n--- PERFORMANCE ---")
    for modelo, perf in analise['performance'].items():
        print(f"\n{modelo.upper()}:")
        print(f"  Tempo total: {perf['total_time_ms']} ms")
        print(f"  Planejamento: {perf['planning_time_ms']:.1f} ms")
        print(f"  Geração: {perf['generation_time_ms']:.1f} ms")
        print(f"  Tamanho resposta: {perf['response_length']} caracteres")

    print("\n--- RESUMO ---")
    print(f"Modelo mais rápido: {analise['summary']['fastest_model']}")
    print(f"Resposta mais longa: {analise['summary']['longest_response']}")
    print(f"Tipo de execução: {analise['summary']['execution_type']}")


def exemplo_4_usando_interface():
    """
    Usar a interface gráfica (MODO NORMAL)
    """
    print("\n" + "="*80)
    print("EXEMPLO 4: INTERFACE GRÁFICA")
    print("="*80)

    # IMPORTANTE: Configure as API keys antes
    os.environ["OPENAI_API_KEY"] = "sua_chave_aqui"
    os.environ["GEMINI_API_KEY"] = "sua_chave_aqui"
    os.environ["DEEPSEEK_API_KEY"] = "sua_chave_aqui"

    cha = openCHA(
        planner_llm=LLMType.GEMINI,
        response_generator_llm=LLMType.GEMINI
    )

    # Abre interface gráfica no browser
    cha.run_with_interface()


def exemplo_5_comparacao_por_criatividade():
    """
    Comparar LLMs em tarefas criativas
    """
    print("\n" + "="*80)
    print("EXEMPLO 5: COMPARAÇÃO COM ALTA CRIATIVIDADE")
    print("="*80)

    cha = openCHA()

    # Tarefa criativa - cada LLM vai usar seu próprio orchestrator
    resultado = cha.run(
        query="Escreva um poema curto sobre inteligência artificial",
        use_multi_llm=True,
        compare_models=['chatgpt', 'gemini', 'deepseek'],
        temperature=1.5,  # Bem criativo
        max_tokens=200
    )

    print(resultado)


def exemplo_6_comparacao_por_velocidade():
    """
    Benchmark de velocidade entre os LLMs
    """
    print("\n" + "="*80)
    print("EXEMPLO 6: BENCHMARK DE VELOCIDADE")
    print("="*80)

    cha = openCHA()

    # Pergunta simples para testar velocidade
    resultado = cha.compare_llm_responses_full(
        query="Qual é a capital da França?",
        models=['chatgpt', 'gemini', 'deepseek'],
        temperature=0,  # Determinístico
        max_tokens=50
    )

    # Extrair apenas tempos
    print("\n--- TEMPOS DE EXECUÇÃO ---")
    for modelo in resultado['times']:
        time_total = resultado['times'][modelo]
        time_planejamento = resultado['planning_times'][modelo]
        time_geracao = resultado['generation_times'][modelo]

        print(f"\n{modelo.upper()}:")
        print(f"  Total: {time_total} ms")
        print(f"  Planejamento: {time_planejamento:.1f} ms ({time_planejamento/time_total*100:.0f}%)")
        print(f"  Geração: {time_geracao:.1f} ms ({time_geracao/time_total*100:.0f}%)")


def exemplo_7_reset_e_cache():
    """
    Gerenciar cache e reset
    """
    print("\n" + "="*80)
    print("EXEMPLO 7: RESET E CACHE")
    print("="*80)

    cha = openCHA(
        multi_llm_enable_cache=True,
        multi_llm_timeout=45
    )

    # Primeira execução (vai para cache)
    print("Primeira execução...")
    resultado1 = cha.run(
        "O que é Python?",
        use_multi_llm=True,
        compare_models=['chatgpt', 'gemini']
    )

    # Segunda execução (pode usar cache)
    print("\nSegunda execução (pode usar cache)...")
    resultado2 = cha.run(
        "O que é Python?",
        use_multi_llm=True,
        compare_models=['chatgpt', 'gemini']
    )

    # Limpar cache
    print("\nLimpando cache...")
    cha.clear_multi_llm_cache()

    # Terceira execução (sem cache)
    print("Terceira execução (sem cache)...")
    resultado3 = cha.run(
        "O que é Python?",
        use_multi_llm=True,
        compare_models=['chatgpt', 'gemini']
    )

    # Reset completo
    print("\nFazendo reset...")
    cha.reset()


def exemplo_8_modelos_disponiveis():
    """
    Verificar modelos disponíveis
    """
    print("\n" + "="*80)
    print("EXEMPLO 8: MODELOS DISPONÍVEIS")
    print("="*80)

    cha = openCHA()

    modelos = cha.get_available_models()
    print(f"\nModelos disponíveis: {modelos}")

    # Comparar apenas alguns
    print("\nComparando apenas ChatGPT e Gemini...")
    resultado = cha.run(
        "teste",
        use_multi_llm=True,
        compare_models=['chatgpt', 'gemini']
    )


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                  openCHA REFATORADO COM ORQUESTRAÇÃO COMPLETA              ║
    ║                                                                            ║
    ║  NOVO: Cada LLM tem seu próprio Orchestrator (pensa + escreve)           ║
    ║  RESULTADO: Comparação mais justa e respostas mais profundas              ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)

    # Escolha qual exemplo rodar
    exemplos = {
        "1": ("Modo Normal (Um LLM)", exemplo_1_modo_normal),
        "2": ("Comparação Completa", exemplo_2_comparacao_completa),
        "3": ("Análise Detalhada", exemplo_3_analise_detalhada),
        "4": ("Interface Gráfica", exemplo_4_usando_interface),
        "5": ("Criatividade", exemplo_5_comparacao_por_criatividade),
        "6": ("Velocidade", exemplo_6_comparacao_por_velocidade),
        "7": ("Reset e Cache", exemplo_7_reset_e_cache),
        "8": ("Modelos Disponíveis", exemplo_8_modelos_disponiveis),
    }

    print("\nEscolha um exemplo para rodar:\n")
    for key, (descricao, _) in exemplos.items():
        print(f"  {key}. {descricao}")

    escolha = input("\nOpção (1-8) [default: 2]: ").strip() or "2"

    if escolha in exemplos:
        nome, funcao = exemplos[escolha]
        print(f"\n\nExecutando: {nome}\n")
        funcao()
    else:
        print("Opção inválida!")
