import gradio as gr
import logging
from typing import List, Tuple, Any, Dict, Optional

logger = logging.getLogger(__name__)


class Interface:
    """
    Gradio UI com duas abas:
      1) Chat normal (single agent)
      2) Comparação Multi-LLM (LLM Arena-style cards)
    """

    def __init__(self):
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
        with self.gr.Blocks(
            theme=gr.themes.Soft(),
            title="openCHA",
            css="""
                .arena-grid { gap: 12px; }
                .arena-card {
                    border: 1px solid rgba(0,0,0,0.08);
                    border-radius: 14px;
                    padding: 12px;
                    background: rgba(255,255,255,0.9);
                    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
                    min-height: 420px;
                }
                .arena-title {
                    font-size: 16px;
                    font-weight: 700;
                    margin-bottom: 6px;
                }
                .arena-meta {
                    font-size: 12px;
                    opacity: 0.8;
                    margin-bottom: 10px;
                    line-height: 1.35;
                }
                .arena-answer {
                    font-size: 14px;
                    line-height: 1.45;
                    white-space: pre-wrap;
                }
                .badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 999px;
                    font-size: 12px;
                    border: 1px solid rgba(0,0,0,0.12);
                    margin-right: 6px;
                }
                .badge-fast { border-color: rgba(34,197,94,0.35); }
                .badge-slow { border-color: rgba(239,68,68,0.35); }
                .small-note { font-size: 12px; opacity: 0.75; }
            """
        ) as demo:

            gr.Markdown(
                """
                # 🔷 openCHA
                **Duas formas de usar:**
                - **Chat normal** (um agente)
                - **Comparação Multi-LLM** (estilo LLM Arena, cards lado a lado)
                """
            )

            # =========================
            # API KEYS (GLOBAL)
            # =========================
            with gr.Accordion("🔑 Configuração de API Keys", open=True):
                gr.Markdown("*Configure suas chaves de API antes de começar*")

                with gr.Row():
                    openai_key = gr.Textbox(
                        label="🟢 OpenAI API Key (ChatGPT)",
                        type="password",
                        value="",
                        placeholder="sk-...",
                    )
                    serp_key = gr.Textbox(
                        label="🔍 SERP API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do SERPAPI",
                    )

                with gr.Row():
                    gemini_key = gr.Textbox(
                        label="🔵 Gemini API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do Google Gemini",
                    )
                    deepseek_key = gr.Textbox(
                        label="🟣 DeepSeek API Key",
                        type="password",
                        value="",
                        placeholder="Sua chave do DeepSeek",
                    )

            # =========================
            # CONFIG DO AGENTE (GLOBAL)
            # =========================
            with gr.Accordion("⚙️ Configurações do Agente", open=False):
                with gr.Row():
                    use_history = gr.Checkbox(
                        label="💬 Usar histórico da conversa",
                        value=True,
                    )
                    tasks_selector = gr.CheckboxGroup(
                        choices=available_tasks,
                        label="🛠️ Ferramentas (Tasks) disponíveis",
                        value=[],
                    )

            gr.Markdown("---")

            # =========================
            # ABAS
            # =========================
            with gr.Tabs():

                # =========================================================
                # ABA 1: CHAT NORMAL
                # =========================================================
                with gr.Tab("💬 Chat normal"):
                    chatbot = gr.Chatbot(
                        label="Conversa",
                        bubble_full_width=False,
                        height=520,
                        show_copy_button=True,
                    )

                    with gr.Row():
                        msg_chat = gr.Textbox(
                            placeholder="Digite sua mensagem...",
                            lines=3,
                            scale=8,
                            show_label=False,
                        )
                        with gr.Column(scale=1, min_width=120):
                            btn_send_chat = gr.Button("🚀 Enviar", variant="primary")
                            btn_upload_chat = gr.UploadButton(
                                "📎 Arquivo",
                                file_types=["text", "pdf", "image", "audio", "video"],
                            )

                    with gr.Row():
                        btn_clear_chat = gr.Button("🗑️ Limpar conversa", variant="secondary")
                        gr.Markdown("<span class='small-note'>Enter envia | Shift+Enter quebra linha</span>")

                    state_chat_history = gr.State([])

                    def render_history(chat_history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
                        return [(u, a) for (u, a) in chat_history if a is not None]

                    def reset_wrapper_chat() -> Tuple[List, List]:
                        try:
                            reset()
                            return [], []
                        except Exception as e:
                            logger.error(f"Erro ao resetar: {e}")
                            return [], []

                    def respond_chat_wrapper(
                        msg: str,
                        openai: str,
                        serp: str,
                        gemini: str,
                        deepseek: str,
                        chat_hist: List[Tuple[str, str]],
                        use_hist: bool,
                        tasks: List[str],
                    ):
                        if not msg or not msg.strip():
                            chat_hist.append((msg, "⚠️ Digite uma mensagem."))
                            return "", chat_hist

                        chat_hist.append((msg, "⏳ Processando..."))
                        yield "", chat_hist

                        try:
                            empty_msg, updated = respond(
                                msg, openai, serp, gemini, deepseek,
                                chat_hist[:-1],
                                use_hist, tasks,
                                False,  # use_multi_llm = False
                                None    # compare_models = None
                            )
                            yield empty_msg, updated
                        except Exception as e:
                            logger.error(f"Erro respond_chat_wrapper: {e}", exc_info=True)
                            chat_hist[-1] = (msg, f"❌ Erro: {str(e)}")
                            yield "", chat_hist

                    # Eventos Chat normal
                    msg_chat.submit(
                        fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key, state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history],
                    )
                    btn_send_chat.click(
                        fn=respond_chat_wrapper,
                        inputs=[msg_chat, openai_key, serp_key, gemini_key, deepseek_key, state_chat_history, use_history, tasks_selector],
                        outputs=[msg_chat, state_chat_history],
                    )
                    state_chat_history.change(
                        fn=render_history,
                        inputs=[state_chat_history],
                        outputs=[chatbot],
                    )
                    btn_clear_chat.click(
                        fn=reset_wrapper_chat,
                        inputs=[],
                        outputs=[state_chat_history, chatbot],
                    )
                    btn_upload_chat.upload(
                        fn=upload_meta,
                        inputs=[state_chat_history, btn_upload_chat],
                        outputs=[state_chat_history],
                    )

                # =========================================================
                # ABA 2: MULTI-LLM ARENA
                # =========================================================
                with gr.Tab("🏟️ Comparação Multi-LLM"):
                    gr.Markdown(
                        """
                        **Modo LLM Arena**: escreva um prompt e veja respostas lado a lado.
                        """
                    )

                    with gr.Row():
                        compare_models = gr.CheckboxGroup(
                            label="🤖 Modelos",
                            choices=["chatgpt", "gemini", "deepseek"],
                            value=["chatgpt", "gemini"],
                            scale=3,
                        )
                        show_metrics = gr.Checkbox(
                            label="📊 Mostrar métricas (tempo/tokens)",
                            value=True,
                            scale=1,
                        )

                    with gr.Row():
                        msg_arena = gr.Textbox(
                            placeholder="Digite a pergunta para comparar os modelos...",
                            lines=3,
                            scale=8,
                            show_label=False,
                        )
                        with gr.Column(scale=1, min_width=140):
                            btn_send_arena = gr.Button("⚔️ Comparar", variant="primary")
                            btn_clear_arena = gr.Button("🧹 Limpar", variant="secondary")

                    # Saídas Arena (cards)
                    with gr.Row(elem_classes="arena-grid"):
                        card_chatgpt = gr.Markdown(elem_classes="arena-card")
                        card_gemini = gr.Markdown(elem_classes="arena-card")
                        card_deepseek = gr.Markdown(elem_classes="arena-card")

                    arena_status = gr.Markdown()

                    def _fmt_ms(x: Optional[float]) -> str:
                        if x is None:
                            return "-"
                        return f"{x:.2f} ms"

                    def _build_card(
                        model_key: str,
                        title: str,
                        response: Optional[str],
                        error: Optional[str],
                        time_ms: Optional[float],
                        planning_ms: Optional[float],
                        gen_ms: Optional[float],
                        is_fastest: bool,
                        show_metrics_flag: bool,
                    ) -> str:
                        badge = ""
                        if show_metrics_flag and time_ms is not None:
                            badge = f"<span class='badge {'badge-fast' if is_fastest else ''}'>⏱ {time_ms:.0f} ms</span>"

                        meta_lines = []
                        if show_metrics_flag:
                            meta_lines.append(f"{badge}")
                            meta_lines.append(f"<span class='badge'>🧠 {planning_ms:.0f} ms</span>" if planning_ms is not None else "<span class='badge'>🧠 -</span>")
                            meta_lines.append(f"<span class='badge'>✍️ {gen_ms:.0f} ms</span>" if gen_ms is not None else "<span class='badge'>✍️ -</span>")

                        meta_html = ""
                        if show_metrics_flag:
                            meta_html = "<div class='arena-meta'>" + " ".join(meta_lines) + "</div>"

                        if error:
                            body = f"❌ **Erro**: {error}"
                        else:
                            body = response or "—"

                        return f"""
<div class="arena-title">{title}</div>
{meta_html}
<div class="arena-answer">{body}</div>
"""

                    def respond_arena(
                        msg: str,
                        openai: str,
                        serp: str,
                        gemini: str,
                        deepseek: str,
                        use_hist: bool,
                        tasks: List[str],
                        models: List[str],
                        show_metrics_flag: bool,
                    ):
                        if not msg or not msg.strip():
                            return (
                                "⚠️ Digite uma mensagem.",
                                "", "", "", ""
                            )

                        if not models:
                            return (
                                "⚠️ Selecione pelo menos 1 modelo.",
                                "", "", "", ""
                            )

                        # (Opcional) valida keys pelos modelos escolhidos
                        missing = []
                        if "chatgpt" in models and not openai:
                            missing.append("OpenAI")
                        if "gemini" in models and not gemini:
                            missing.append("Gemini")
                        if "deepseek" in models and not deepseek:
                            missing.append("DeepSeek")
                        if missing:
                            return (
                                f"⚠️ API Key faltando: {', '.join(missing)}",
                                "", "", "", ""
                            )

                        status = f"⏳ Comparando {', '.join(models)}..."
                        yield status, "", "", ""

                        # Chama seu respond() em modo multi
                        empty_msg, updated_hist = respond(
                            msg, openai, serp, gemini, deepseek,
                            [],                 # arena não precisa do histórico visual aqui
                            use_hist, tasks,
                            True,               # use_multi_llm = True
                            models              # compare_models
                        )

                        # O respond() no seu openCHA coloca um item no chat_history com a string formatada.
                        # Vamos pegar a última resposta do agente (texto do relatório) e também tentar extrair “results”.
                        # Como você já formata o relatório em texto, aqui a gente só mostra esse texto por modelo
                        # usando um parse simples por blocos.
                        report_text = ""
                        if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                            report_text = updated_hist[-1][1] or ""

                        # Parse bem simples do seu relatório
                        # Espera blocos: "🤖 CHATGPT" ... "📝 Resposta:" ...
                        def extract_block(model_upper: str) -> str:
                            if not report_text:
                                return ""
                            key = f"🤖 {model_upper}"
                            idx = report_text.find(key)
                            if idx == -1:
                                return ""
                            nxt = report_text.find("🤖 ", idx + 1)
                            block = report_text[idx:] if nxt == -1 else report_text[idx:nxt]
                            # pega só depois de "📝 Resposta:"
                            marker = "📝 Resposta:"
                            m = block.find(marker)
                            if m != -1:
                                return block[m + len(marker):].strip()
                            return block.strip()

                        # Extrair tempos (também parse simples)
                        def extract_time(model_upper: str, label: str) -> Optional[float]:
                            # label: "Tempo total", "Planejamento", "Geração"
                            # procura dentro do bloco do modelo
                            if not report_text:
                                return None
                            key = f"🤖 {model_upper}"
                            idx = report_text.find(key)
                            if idx == -1:
                                return None
                            nxt = report_text.find("🤖 ", idx + 1)
                            block = report_text[idx:] if nxt == -1 else report_text[idx:nxt]

                            # exemplos de linhas:
                            # "⏱️ Tempo total: 14776.28 ms"
                            # "├─ 🧠 Planejamento: 5910.5 ms"
                            # "└─ ✍️ Geração: 8865.8 ms"
                            mapping = {
                                "Tempo total": "⏱️ Tempo total:",
                                "Planejamento": "Planejamento:",
                                "Geração": "Geração:"
                            }
                            prefix = mapping.get(label)
                            if not prefix:
                                return None

                            for line in block.splitlines():
                                line = line.strip()
                                if prefix in line:
                                    # extrai o número antes de "ms"
                                    try:
                                        num = line.split(prefix, 1)[1].strip()
                                        num = num.replace("ms", "").strip()
                                        return float(num)
                                    except Exception:
                                        return None
                            return None

                        # respostas
                        r_chatgpt = extract_block("CHATGPT") if "chatgpt" in models else "—"
                        r_gemini = extract_block("GEMINI") if "gemini" in models else "—"
                        r_deepseek = extract_block("DEEPSEEK") if "deepseek" in models else "—"

                        # tempos
                        t_chatgpt = extract_time("CHATGPT", "Tempo total") if "chatgpt" in models else None
                        p_chatgpt = extract_time("CHATGPT", "Planejamento") if "chatgpt" in models else None
                        g_chatgpt = extract_time("CHATGPT", "Geração") if "chatgpt" in models else None

                        t_gemini = extract_time("GEMINI", "Tempo total") if "gemini" in models else None
                        p_gemini = extract_time("GEMINI", "Planejamento") if "gemini" in models else None
                        g_gemini = extract_time("GEMINI", "Geração") if "gemini" in models else None

                        t_deepseek = extract_time("DEEPSEEK", "Tempo total") if "deepseek" in models else None
                        p_deepseek = extract_time("DEEPSEEK", "Planejamento") if "deepseek" in models else None
                        g_deepseek = extract_time("DEEPSEEK", "Geração") if "deepseek" in models else None

                        # mais rápido
                        times_map = {
                            "chatgpt": t_chatgpt,
                            "gemini": t_gemini,
                            "deepseek": t_deepseek
                        }
                        valid = {k: v for k, v in times_map.items() if isinstance(v, (int, float))}
                        fastest_key = min(valid, key=valid.get) if valid else None

                        # cards
                        card1 = _build_card(
                            "chatgpt", "ChatGPT",
                            r_chatgpt if "chatgpt" in models else "—",
                            None,
                            t_chatgpt, p_chatgpt, g_chatgpt,
                            fastest_key == "chatgpt",
                            show_metrics_flag,
                        )
                        card2 = _build_card(
                            "gemini", "Gemini",
                            r_gemini if "gemini" in models else "—",
                            None,
                            t_gemini, p_gemini, g_gemini,
                            fastest_key == "gemini",
                            show_metrics_flag,
                        )
                        card3 = _build_card(
                            "deepseek", "DeepSeek",
                            r_deepseek if "deepseek" in models else "—",
                            None,
                            t_deepseek, p_deepseek, g_deepseek,
                            fastest_key == "deepseek",
                            show_metrics_flag,
                        )

                        status_done = "✅ Pronto. Compare as respostas nos cards."
                        yield status_done, card1, card2, card3

                    def clear_arena():
                        return "", "", "", "", ""

                    # Eventos Arena
                    msg_arena.submit(
                        fn=respond_arena,
                        inputs=[msg_arena, openai_key, serp_key, gemini_key, deepseek_key, use_history, tasks_selector, compare_models, show_metrics],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek],
                    )
                    btn_send_arena.click(
                        fn=respond_arena,
                        inputs=[msg_arena, openai_key, serp_key, gemini_key, deepseek_key, use_history, tasks_selector, compare_models, show_metrics],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek],
                    )
                    btn_clear_arena.click(
                        fn=clear_arena,
                        inputs=[],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek, msg_arena],
                    )

            gr.Markdown(
                """
                ---
                <div style='text-align:center; opacity:0.75; font-size:12px;'>
                  openCHA • Chat normal + Comparação Multi-LLM (Arena)
                </div>
                """
            )

        logger.info(f"Lançando interface na porta {server_port}...")
        demo.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            show_error=True,
        )
        logger.info("Interface lançada com sucesso!")

    def close(self):
        logger.info("Fechando interface...")
