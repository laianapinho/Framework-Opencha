# openCHA/interface/base.py
import gradio as gr
import logging
from typing import List, Tuple, Optional, Any

from openCHA.benchmark_ui_helpers import (
    load_dataset_from_gradio_file,
    run_json_benchmark,
    extract_model_response_from_report,
)
from openCHA.benchmark_evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


class Interface:
    """
    Gradio UI com 3 abas:
      1) Chat normal (single agent)
      2) Comparação Multi-LLM (Arena) — cards lado a lado
      3) Benchmark JSON (upload + escolher N + rodar automático)

    Correções aplicadas:
      - Arena: cards LADO A LADO (Row + Columns)
      - Arena: usa extract_model_response_from_report (função correta)
      - Benchmark: usa run_single_question wrapper (compatível com assinatura real do respond)
      - Benchmark: tabela compatível com headers + ok_* (para CLOSED)
    """

    def __init__(self):
        self.gr = gr
        self._benchmark_loader = None
        self._benchmark_info = None
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
                .arena-answer {
                    font-size: 14px;
                    line-height: 1.45;
                    white-space: pre-wrap;
                }
                .small-note { font-size: 12px; opacity: 0.75; }
            """,
        ) as demo:

            gr.Markdown(
                """
                # 🔷 openCHA
                **Modos:**
                - **Chat normal**
                - **Arena Multi-LLM**
                - **Benchmark JSON** (upload + escolher quantidade)
                """
            )

            # =========================
            # API KEYS (GLOBAL)
            # =========================
            with gr.Accordion("🔑 Configuração de API Keys", open=True):
                with gr.Row():
                    openai_key = gr.Textbox(label="🟢 OpenAI API Key", type="password")
                    serp_key = gr.Textbox(label="🔍 SERP API Key", type="password")
                with gr.Row():
                    gemini_key = gr.Textbox(label="🔵 Gemini API Key", type="password")
                    deepseek_key = gr.Textbox(label="🟣 DeepSeek API Key", type="password")

            # =========================
            # CONFIG DO AGENTE (GLOBAL)
            # =========================
            with gr.Accordion("⚙️ Configurações do Agente", open=False):
                with gr.Row():
                    use_history = gr.Checkbox(label="💬 Usar histórico", value=True)
                    tasks_selector = gr.CheckboxGroup(
                        choices=available_tasks, label="🛠️ Tasks", value=[]
                    )

            gr.Markdown("---")

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
                        # Se quiser remover warning no futuro:
                        # type="messages"
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

                    def reset_wrapper_chat():
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
                                False,  # use_multi_llm
                                None
                            )
                            yield empty_msg, updated
                        except Exception as e:
                            logger.error(f"Erro respond_chat_wrapper: {e}", exc_info=True)
                            chat_hist[-1] = (msg, f"❌ Erro: {str(e)}")
                            yield "", chat_hist

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
                # ABA 2: MULTI-LLM ARENA (LADO A LADO)
                # =========================================================
                with gr.Tab("🏟️ Comparação Multi-LLM"):
                    gr.Markdown("**Escreva um prompt e veja respostas lado a lado.**")

                    compare_models = gr.CheckboxGroup(
                        label="🤖 Modelos",
                        choices=["chatgpt", "gemini", "deepseek"],
                        value=["chatgpt", "gemini"],
                    )

                    msg_arena = gr.Textbox(placeholder="Digite a pergunta...", lines=3, show_label=False)
                    btn_send_arena = gr.Button("⚔️ Comparar", variant="primary")
                    arena_status = gr.Markdown()

                    # ✅ lado a lado
                    with gr.Row(equal_height=True, elem_classes="arena-grid"):
                        with gr.Column(scale=1, min_width=320):
                            card_chatgpt = gr.Markdown(elem_classes="arena-card")
                        with gr.Column(scale=1, min_width=320):
                            card_gemini = gr.Markdown(elem_classes="arena-card")
                        with gr.Column(scale=1, min_width=320):
                            card_deepseek = gr.Markdown(elem_classes="arena-card")

                    def respond_arena(msg, openai, serp, gemini, deepseek, use_hist, tasks, models):
                        if not msg or not msg.strip():
                            return "⚠️ Digite uma mensagem.", "", "", ""

                        if not models:
                            return "⚠️ Selecione pelo menos 1 modelo.", "", "", ""

                        yield f"⏳ Comparando {', '.join(models)}...", "", "", ""

                        _, updated_hist = respond(
                            msg, openai, serp, gemini, deepseek,
                            [],
                            use_hist, tasks,
                            True,
                            models
                        )

                        report_text = ""
                        if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                            report_text = updated_hist[-1][1] or ""

                        def pick(model_key: str) -> str:
                            # relatório geralmente usa CHATGPT/GEMINI/DEEPSEEK
                            return extract_model_response_from_report(report_text, model_key.upper()) or "—"

                        out_cg = pick("chatgpt") if "chatgpt" in models else "—"
                        out_gm = pick("gemini") if "gemini" in models else "—"
                        out_ds = pick("deepseek") if "deepseek" in models else "—"

                        yield (
                            "✅ Pronto.",
                            f"### ChatGPT\n\n{out_cg}",
                            f"### Gemini\n\n{out_gm}",
                            f"### DeepSeek\n\n{out_ds}",
                        )

                    btn_send_arena.click(
                        fn=respond_arena,
                        inputs=[msg_arena, openai_key, serp_key, gemini_key, deepseek_key, use_history, tasks_selector, compare_models],
                        outputs=[arena_status, card_chatgpt, card_gemini, card_deepseek],
                    )

                # =========================================================
                # ABA 3: BENCHMARK JSON (N questões)
                # =========================================================
                with gr.Tab("📁 Benchmark JSON"):
                    gr.Markdown(
                        """
                        **Upload de JSON** → escolha **quantidade de questões** → **rodar automático**
                        (sem selecionar uma a uma)
                        """
                    )

                    file_json = gr.File(label="Envie o JSON", file_types=[".json"], type="binary")
                    btn_load = gr.Button("✅ Carregar JSON", variant="primary")
                    load_status = gr.Markdown()

                    models_bench = gr.CheckboxGroup(
                        label="🤖 Modelos",
                        choices=["chatgpt", "gemini", "deepseek"],
                        value=["chatgpt", "gemini"],
                    )

                    num_samples = gr.Slider(
                        minimum=1,
                        maximum=500,
                        value=10,
                        step=1,
                        label="Quantidade de questões",
                    )

                    random_sample = gr.Checkbox(label="🎲 Seleção aleatória", value=False)

                    btn_run = gr.Button("🚀 Rodar Benchmark", variant="primary")
                    bench_report = gr.Textbox(label="Relatório", lines=25)

                    bench_table = gr.Dataframe(
                        headers=[
                            "id", "expected",
                            "chatgpt", "gemini", "deepseek",
                            "ok_chatgpt", "ok_gemini", "ok_deepseek"
                        ],
                        datatype=["str"] * 8,
                        row_count=5,
                        col_count=(8, "fixed"),
                        wrap=True,
                    )

                    def do_load(file_obj):
                        try:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info = info
                            stats = info["stats"]
                            msg = (
                                f"✅ **Carregado!**\n\n"
                                f"• Total: **{stats.get('total_items')}**\n"
                                f"• Tipo: **{info.get('dataset_type', '').upper()}**\n"
                                f"• Campos: pergunta=`{info['mapping'].get('question_field')}`, "
                                f"resposta=`{info['mapping'].get('answer_field')}`\n"
                            )
                            return msg
                        except Exception as e:
                            logger.error(f"Erro load JSON: {e}", exc_info=True)
                            return f"❌ Erro ao carregar JSON: {e}"

                    def do_run(file_obj, models, n, rnd, openai, serp, gemini, deepseek, use_hist, tasks):
                        # garante loader
                        if self._benchmark_loader is None:
                            loader, info = load_dataset_from_gradio_file(file_obj)
                            self._benchmark_loader = loader
                            self._benchmark_info = info

                        # wrapper compatível com engine do benchmark e com assinatura real do respond()
                        def run_single_question(
                            question: str,
                            use_multi_llm: bool = True,
                            compare_models: Optional[List[str]] = None
                        ) -> str:
                            _empty, updated_hist = respond(
                                question,
                                openai, serp, gemini, deepseek,
                                [],          # sem histórico no benchmark (evita contaminação)
                                use_hist,
                                tasks,
                                True,        # força multi-llm
                                compare_models,
                            )
                            if updated_hist and updated_hist[-1] and len(updated_hist[-1]) == 2:
                                return updated_hist[-1][1] or ""
                            return ""

                        report, rows = run_json_benchmark(
                            run_single_question=run_single_question,
                            loader=self._benchmark_loader,
                            models=models,
                            num_samples=int(n),
                            show_per_question=True,
                        )

                        # tabela com colunas ok_* quando dataset_type == closed
                        evaluator = BenchmarkEvaluator()

                        def ok_flag(dataset_type: str, expected: Any, answer: str) -> str:
                            if (dataset_type or "").lower().strip() != "closed":
                                return ""
                            exp = str(expected).strip().lower()
                            pred = evaluator.extract_answer(answer)
                            return "✅" if pred == exp else "❌"

                        table_rows: List[List[str]] = []
                        for r in rows:
                            ds_type = str(r.get("dataset_type", ""))
                            expected = r.get("expected", "")
                            cg = r.get("chatgpt", "")
                            gm = r.get("gemini", "")
                            ds = r.get("deepseek", "")

                            table_rows.append([
                                str(r.get("id", "")),
                                str(expected),
                                str(cg),
                                str(gm),
                                str(ds),
                                ok_flag(ds_type, expected, str(cg)),
                                ok_flag(ds_type, expected, str(gm)),
                                ok_flag(ds_type, expected, str(ds)),
                            ])

                        return report, table_rows

                    btn_load.click(fn=do_load, inputs=[file_json], outputs=[load_status])

                    btn_run.click(
                        fn=do_run,
                        inputs=[
                            file_json, models_bench, num_samples, random_sample,
                            openai_key, serp_key, gemini_key, deepseek_key,
                            use_history, tasks_selector
                        ],
                        outputs=[bench_report, bench_table],
                    )

            demo.launch(
                share=share,
                server_port=server_port,
                server_name="0.0.0.0",
                show_error=True,
            )
