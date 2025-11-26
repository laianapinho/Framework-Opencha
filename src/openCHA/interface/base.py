import gradio as gr
import logging
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)


class Interface:
    """
    Interface gráfica com Gradio para openCHA com suporte completo a Multi-LLM.
    
    Recursos:
        - Chat interativo com histórico
        - Upload de múltiplos tipos de arquivo
        - Seleção de tarefas (tasks) disponíveis
        - Configuração de API keys (OpenAI, SERP, Gemini, DeepSeek)
        - Modo Multi-LLM para comparação entre modelos
        - Seleção flexível de modelos a comparar
    """

    def __init__(self):
        """Inicializa a interface."""
        self.gr = gr
        logger.info("Interface inicializada")

    def prepare_interface(
        self,
        respond,
        reset,
        upload_meta,
        available_tasks: List[str],
        share: bool = False,
        server_port: int = 7860,
    ):
        """
        Configura e lança a interface gráfica completa.

        Args:
            respond: Função callback para processar mensagens do usuário
            reset: Função callback para resetar o estado do agente
            upload_meta: Função callback para processar uploads de arquivos
            available_tasks: Lista de tarefas (tools) disponíveis
            share: Se True, cria link público via Gradio (padrão: False)
            server_port: Porta do servidor local (padrão: 7860)
        """
        with self.gr.Blocks(
            theme=gr.themes.Soft(),
            title="openCHA + Multi-LLM",
            css="""
                .message-row {margin: 8px 0;}
                .model-section {background: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;}
            """
        ) as demo:
            
            # ========================================
            # CABEÇALHO
            # ========================================
            gr.Markdown(
                """
                # 🔷 openCHA + Multi-LLM
                ### Sistema Inteligente com Comparação de Modelos (ChatGPT | Gemini | DeepSeek)
                
                **Modos de Uso:**
                - 🤖 **Modo Normal**: Um agente com orquestração completa
                - 🌐 **Modo Multi-LLM**: Compare respostas de múltiplos modelos lado a lado
                """
            )

            # ========================================
            # CONFIGURAÇÃO DE API KEYS
            # ========================================
            with gr.Accordion("🔑 Configuração de API Keys", open=True):
                gr.Markdown("*Configure suas chaves de API antes de começar*")
                
                with gr.Row():
                    openai_key = gr.Textbox(
                        label="🟢 OpenAI API Key (ChatGPT)",
                        type="password",
                        value="",
                        placeholder="sk-...",
                        info="Necessária para usar ChatGPT"
                    )
                    serp_key = gr.Textbox(
                        label="🔍 SERP API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do SERPAPI",
                        info="Para buscas na web"
                    )
                
                with gr.Row():
                    gemini_key = gr.Textbox(
                        label="🔵 Gemini API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do Google Gemini",
                        info="Necessária para usar Gemini"
                    )
                    deepseek_key = gr.Textbox(
                        label="🟣 DeepSeek API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do DeepSeek",
                        info="Necessária para usar DeepSeek"
                    )

            # ========================================
            # CONFIGURAÇÕES DO AGENTE
            # ========================================
            with gr.Accordion("⚙️ Configurações do Agente", open=False):
                with gr.Row():
                    use_history = gr.Checkbox(
                        label="💬 Usar histórico da conversa",
                        value=True,
                        info="Se marcado, o agente lembrará do contexto anterior"
                    )
                    
                    tasks_selector = gr.CheckboxGroup(
                        choices=available_tasks,
                        label="🛠️ Ferramentas (Tasks) Disponíveis",
                        value=[],
                        info="Selecione as ferramentas que o agente pode usar"
                    )

            # ========================================
            # CONFIGURAÇÕES MULTI-LLM
            # ========================================
            with gr.Accordion("🌐 Modo Multi-LLM (Comparação de Modelos)", open=False):
                gr.Markdown(
                    """
                    **Ative este modo para comparar respostas de múltiplos modelos simultaneamente.**
                    
                    ⚠️ *Atenção*: O modo Multi-LLM consome mais tokens e pode demorar mais.
                    Cada modelo executa sua própria orquestração completa (planejamento + geração).
                    """
                )
                
                with gr.Row():
                    use_multi_llm = gr.Checkbox(
                        label="✅ Ativar Comparação Multi-LLM",
                        value=False,
                        info="Marque para comparar múltiplos modelos"
                    )
                
                with gr.Row():
                    compare_models = gr.CheckboxGroup(
                        label="🤖 Modelos para Comparar",
                        choices=["chatgpt", "gemini", "deepseek"],
                        value=["chatgpt", "gemini"],
                        info="Selecione quais modelos deseja comparar (mínimo 2 recomendado)"
                    )
                
                with gr.Row():
                    gr.Markdown(
                        """
                        **💡 Dicas:**
                        - Para consultas rápidas: Use apenas 2 modelos
                        - Para análise completa: Use todos os 3 modelos
                        - O tempo de resposta será do modelo mais lento
                        """
                    )

            # ========================================
            # ÁREA DE CHAT
            # ========================================
            gr.Markdown("---")
            
            chatbot = gr.Chatbot(
                label="💬 Conversa com openCHA",
                bubble_full_width=False,
                height=450,
                show_copy_button=True,
                avatar_images=(
                    None,  # Avatar do usuário
                    "https://raw.githubusercontent.com/gradio-app/gradio/main/js/chatbot/bot.svg"  # Avatar do bot
                )
            )

            # ========================================
            # ÁREA DE INPUT
            # ========================================
            with gr.Row():
                message = gr.Textbox(
                    label="📝 Sua Mensagem",
                    placeholder="Digite sua pergunta ou comando aqui...",
                    lines=3,
                    scale=8,
                    autofocus=True,
                    show_label=False
                )
                
                with gr.Column(scale=1, min_width=100):
                    send_btn = gr.Button("🚀 Enviar", variant="primary", size="lg")
                    upload_btn = gr.UploadButton(
                        "📎 Arquivo",
                        file_types=["text", "pdf", "image", "audio", "video"],
                        size="sm"
                    )

            # ========================================
            # BOTÕES DE CONTROLE
            # ========================================
            with gr.Row():
                clear_btn = gr.Button("🗑️ Limpar Conversa", variant="secondary")
                
                gr.Markdown(
                    """
                    <div style='text-align: right; color: #666; font-size: 12px;'>
                    💡 Pressione Enter para enviar | Shift+Enter para nova linha
                    </div>
                    """,
                    elem_classes="message-row"
                )

            # ========================================
            # ESTADO INTERNO
            # ========================================
            state_chat_history = gr.State([])

            # ========================================
            # FUNÇÕES AUXILIARES
            # ========================================
            def render_history(chat_history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
                """
                Renderiza o histórico do chat filtrando valores None.
                
                Args:
                    chat_history: Lista de tuplas (user_msg, bot_msg)
                
                Returns:
                    Lista filtrada para exibição no Chatbot
                """
                return [(u, a) for (u, a) in chat_history if a is not None]

            def reset_wrapper() -> Tuple[List, List]:
                """
                Wrapper para a função reset que limpa tanto o estado quanto a UI.
                
                Returns:
                    Tupla (chat_history_vazio, chatbot_vazio)
                """
                try:
                    reset()  # Chama a função de reset do openCHA
                    logger.info("Estado resetado pela interface")
                    return [], []
                except Exception as e:
                    logger.error(f"Erro ao resetar: {e}")
                    return [], []

            def validate_inputs(
                msg: str,
                use_multi: bool,
                models: List[str],
                openai: str,
                gemini: str,
                deepseek: str
            ) -> Tuple[bool, str]:
                """
                Valida as entradas do usuário antes de processar.
                
                Args:
                    msg: Mensagem do usuário
                    use_multi: Se modo multi-LLM está ativo
                    models: Lista de modelos selecionados
                    openai: API key OpenAI
                    gemini: API key Gemini
                    deepseek: API key DeepSeek
                
                Returns:
                    Tupla (is_valid, error_message)
                """
                # Valida mensagem vazia
                if not msg or not msg.strip():
                    return False, "⚠️ Por favor, digite uma mensagem antes de enviar."
                
                # Valida modo Multi-LLM
                if use_multi:
                    if not models or len(models) == 0:
                        return False, "⚠️ Modo Multi-LLM ativado, mas nenhum modelo foi selecionado. Selecione pelo menos um modelo."
                    
                    # Valida API keys dos modelos selecionados
                    missing_keys = []
                    if "chatgpt" in models and not openai:
                        missing_keys.append("OpenAI")
                    if "gemini" in models and not gemini:
                        missing_keys.append("Gemini")
                    if "deepseek" in models and not deepseek:
                        missing_keys.append("DeepSeek")
                    
                    if missing_keys:
                        return False, f"⚠️ API Keys faltando para: {', '.join(missing_keys)}. Configure-as antes de usar estes modelos."
                
                return True, ""

            def respond_wrapper(
                msg: str,
                openai: str,
                serp: str,
                gemini: str,
                deepseek: str,
                chat_hist: List[Tuple[str, str]],
                use_hist: bool,
                tasks: List[str],
                use_multi: bool,
                models: List[str]
            ) -> Tuple[str, List[Tuple[str, str]]]:
                """
                Wrapper para a função respond que adiciona validação e tratamento de erros.
                
                Returns:
                    Tupla (mensagem_limpa, chat_history_atualizado)
                """
                # Validação de inputs
                is_valid, error_msg = validate_inputs(
                    msg, use_multi, models, openai, gemini, deepseek
                )
                
                if not is_valid:
                    chat_hist.append((msg, error_msg))
                    return "", chat_hist
                
                # Adiciona mensagem temporária de processamento
                if use_multi:
                    processing_msg = f"⏳ Comparando respostas de {len(models)} modelo(s): {', '.join(models)}..."
                else:
                    processing_msg = "⏳ Processando sua mensagem..."
                
                chat_hist.append((msg, processing_msg))
                yield "", chat_hist  # Atualização intermediária
                
                # Chama a função respond real
                try:
                    empty_msg, updated_hist = respond(
                        msg, openai, serp, gemini, deepseek,
                        chat_hist[:-1],  # Remove mensagem temporária
                        use_hist, tasks, use_multi, models
                    )
                    
                    yield empty_msg, updated_hist
                    
                except Exception as e:
                    logger.error(f"Erro em respond_wrapper: {e}", exc_info=True)
                    chat_hist[-1] = (msg, f"❌ Erro ao processar: {str(e)}")
                    yield "", chat_hist

            # ========================================
            # CONEXÃO DE EVENTOS
            # ========================================
            
            # Envio via Enter (textbox.submit)
            message.submit(
                fn=respond_wrapper,
                inputs=[
                    message,
                    openai_key,
                    serp_key,
                    gemini_key,
                    deepseek_key,
                    state_chat_history,
                    use_history,
                    tasks_selector,
                    use_multi_llm,
                    compare_models,
                ],
                outputs=[message, state_chat_history],
            )

            # Envio via botão
            send_btn.click(
                fn=respond_wrapper,
                inputs=[
                    message,
                    openai_key,
                    serp_key,
                    gemini_key,
                    deepseek_key,
                    state_chat_history,
                    use_history,
                    tasks_selector,
                    use_multi_llm,
                    compare_models,
                ],
                outputs=[message, state_chat_history],
            )

            # Atualização do chatbot quando o estado muda
            state_chat_history.change(
                fn=render_history,
                inputs=[state_chat_history],
                outputs=[chatbot],
            )

            # Botão de limpar
            clear_btn.click(
                fn=reset_wrapper,
                inputs=[],
                outputs=[state_chat_history, chatbot],
            )

            # Upload de arquivos
            upload_btn.upload(
                fn=upload_meta,
                inputs=[state_chat_history, upload_btn],
                outputs=[state_chat_history],
            )

            # ========================================
            # EXEMPLOS (OPCIONAL)
            # ========================================
            gr.Markdown("---")
            with gr.Accordion("💡 Exemplos de Uso", open=False):
                gr.Examples(
                    examples=[
                        ["Explique o que é Machine Learning em termos simples"],
                        ["Qual a diferença entre IA, Machine Learning e Deep Learning?"],
                        ["Crie um plano de estudos para aprender Python em 3 meses"],
                        ["Quais são as tendências em IA para 2025?"],
                        ["Compare os prós e contras de usar ChatGPT vs Gemini"],
                    ],
                    inputs=message,
                    label="Clique em um exemplo para testar"
                )

            # ========================================
            # RODAPÉ
            # ========================================
            gr.Markdown(
                """
                ---
                <div style='text-align: center; color: #666; font-size: 12px;'>
                    <p>🔷 <b>openCHA + Multi-LLM</b> | Powered by Tree of Thought Orchestration</p>
                    <p>⚡ Suporta ChatGPT, Gemini e DeepSeek | 🛠️ Ferramentas extensíveis</p>
                </div>
                """,
                elem_classes="message-row"
            )

        # ========================================
        # LANÇAMENTO DO SERVIDOR
        # ========================================
        logger.info(f"Lançando interface na porta {server_port}...")
        demo.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",  # Permite acesso externo
            show_error=True,
            # inbrowser=True,  # Descomente para abrir automaticamente no navegador
        )
        
        logger.info("Interface lançada com sucesso!")

    def close(self):
        """
        Fecha a interface (útil para testes ou reinicios).
        """
        logger.info("Fechando interface...")
        # Gradio fecha automaticamente ao finalizar o script
