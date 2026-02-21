#!/usr/bin/env python3
"""
Script de teste para verificar se ChatGPT, Gemini e DeepSeek funcionam
Execute no terminal para diagnosticar o problema
"""

import os
import sys

print("=" * 80)
print("🧪 TESTE DE LLMs - Verificar se funcionam")
print("=" * 80)

# TESTE 1: Verificar se API keys existem
print("\n1️⃣ VERIFICANDO API KEYS...")
print("-" * 80)

openai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

print(f"OPENAI_API_KEY: {openai_key[:10]}..." if openai_key else "OPENAI_API_KEY: ❌ NÃO CONFIGURADA")
print(f"GEMINI_API_KEY: {gemini_key[:10]}..." if gemini_key else "GEMINI_API_KEY: ❌ NÃO CONFIGURADA")
print(f"DEEPSEEK_API_KEY: {deepseek_key[:10]}..." if deepseek_key else "DEEPSEEK_API_KEY: ❌ NÃO CONFIGURADA")

if not openai_key or not deepseek_key:
    print("\n❌ ERRO: API keys não estão configuradas!")
    print("\nConfigure assim:")
    print('  export OPENAI_API_KEY="sua-chave-aqui"')
    print('  export DEEPSEEK_API_KEY="sua-chave-aqui"')
    print('  export GEMINI_API_KEY="sua-chave-aqui"')
    sys.exit(1)

print("✅ API keys encontradas!")

# TESTE 2: Importar as classes
print("\n2️⃣ IMPORTANDO CLASSES...")
print("-" * 80)

try:
    from openCHA.llms import initialize_llm, LLMType
    print("✅ Importação bem-sucedida")
except Exception as e:
    print(f"❌ Erro ao importar: {e}")
    sys.exit(1)

# TESTE 3: Testar cada modelo individualmente
print("\n3️⃣ TESTANDO CADA MODELO...")
print("-" * 80)

modelos = {
    'chatgpt': LLMType.OPENAI,
    'deepseek': LLMType.DEEPSEEK,
    'gemini': LLMType.GEMINI,
}

test_query = "What are the main symptoms of cancer?"
resultados = {}

for nome, llm_type in modelos.items():
    print(f"\n▶️  Testando {nome.upper()}...")

    try:
        # ETAPA 1: Criar instância
        print(f"   ├─ Criando instância...", end=" ")
        llm = initialize_llm(llm_type)
        print("✅")

        # ETAPA 2: Testar geração
        print(f"   ├─ Gerando resposta...", end=" ")
        response = llm.generate(
            test_query,
            max_tokens=50,
            temperature=0
        )
        print("✅")

        # ETAPA 3: Validar resposta
        print(f"   ├─ Validando resposta...", end=" ")
        if response and isinstance(response, str) and len(response.strip()) > 5:
            print("✅")
            print(f"   └─ 📝 Resposta: {response[:80]}...")
            resultados[nome] = "✅ FUNCIONANDO"
        else:
            print("❌")
            print(f"   └─ 📝 Resposta vazia ou inválida: {response!r}")
            resultados[nome] = "⚠️ RESPOSTA VAZIA"

    except Exception as e:
        print("❌")
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"   └─ ❌ {error_type}: {error_msg}")
        resultados[nome] = f"❌ {error_type}"

# TESTE 4: Resumo
print("\n" + "=" * 80)
print("4️⃣ RESUMO DOS RESULTADOS")
print("=" * 80)

for nome, status in resultados.items():
    print(f"{nome.upper():<15} {status}")

print("\n" + "=" * 80)

# Análise
funcionando = sum(1 for s in resultados.values() if "✅" in s)
total = len(resultados)

print(f"\n📊 {funcionando}/{total} modelos funcionando")

if funcionando == 0:
    print("\n❌ PROBLEMA: Nenhum modelo funciona!")
    print("\nPossíveis causas:")
    print("  1. API keys inválidas")
    print("  2. Problema de conexão de rede")
    print("  3. Quota excedida")
    print("  4. Firewall bloqueando")
elif funcionando < total:
    print(f"\n⚠️ AVISO: Apenas {funcionando} modelo(s) funcionando")
    print("   Os outros precisam ser investigados")
else:
    print("\n✅ SUCESSO: Todos os modelos funcionam!")

print("\n" + "=" * 80)
