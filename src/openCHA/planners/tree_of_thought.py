"""
TreeOfThoughtPlanner com Restrição a Biologia + Saúde
Gera 3 estratégias paralelas para responder perguntas

✅ MELHORIAS:
   - Restrição a Biologia + Saúde (zero-shot classification)
   - 3 estratégias paralelas visíveis no terminal
   - Sem código Python desnecessário
   - Planejamento eficiente
   - Recusa inteligente para tópicos fora do escopo
"""
import re
import logging
from typing import Any, List

from openCHA.planners import BasePlanner

logger = logging.getLogger(__name__)


class TreeOfThoughtPlanner(BasePlanner):
    """
    Tree of Thought Planner com restrição a Biologia + Saúde.

    ✅ Características:
    - Restrição de domínio: APENAS Biologia e Saúde
    - 3 estratégias paralelas geradas durante execução
    - Sistema de zero-shot classification para decidir se é biologia/saúde
    - Resposta final limpa (sem código Python)
    - Logging estruturado
    - Rejeição educada para tópicos fora do escopo
    """

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000
    restrict_to_biology_health: bool = True  # ✅ ATIVO - Domínio restrito

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "tree-of-thought-biology-health-planner"

    @property
    def _planner_model(self):
        return self.llm_model

    @property
    def _response_generator_model(self):
        return self.llm_model

    @property
    def _stop(self) -> List[str]:
        return ["Wait", "---"]

    @property
    def _planner_prompt(self):
        """
        ✅ PROMPT COM ZERO-SHOT CLASSIFICATION:
        - Instrui o modelo a classificar se é biologia/saúde
        - Se não é, deve responder com "REFUSE:"
        - Se é, gera 3 estratégias paralelas
        - O LLM decide inteligentemente
        """
        return [
            """You are a knowledgeable biology and health assistant. Your role is STRICTLY limited to biology and health-related topics.

IMPORTANT: First, determine if the question is about biology, medicine, health, wellness, nutrition, fitness, mental health, genetics, biochemistry, anatomy, physiology, pathology, or medical conditions.

If the question is NOT related to any of these biology and health topics:
- Respond ONLY with: "REFUSE: Not a biology or health-related question."
- Do NOT provide any information about non-biology/non-health topics
- Do NOT try to connect unrelated topics to biology or health

If the question IS biology or health-related:
- Generate 3 parallel strategies to answer it
- You MUST provide your response in this EXACT format:

Strategy 1: [First approach to answer the question]
Strategy 2: [Second different approach to answer the question]
Strategy 3: [Third different approach to answer the question]

Best Strategy: [Which strategy is best and why]

Final Answer: [Your direct answer based on the best strategy]

IMPORTANT:
- Do NOT generate Python code, function calls, or tool descriptions
- Keep your answer in the same language as the question
- Keep strategies brief but informative

Question: {input}

Response:"""
        ]

    @property
    def _shorten_prompt(self):
        return (
            "Summarize the following text. Make sure to keep the main ideas "
            "and objectives in the summary. Keep the links "
            "exactly as they are: "
            "{chunk}"
        )

    def task_descriptions(self):
        """
        ✅ COMPATIBILIDADE COM BasePlanner
        Mantido para compatibilidade, mas não usado neste planner otimizado
        """
        return "".join(
            [
                (
                    "\n-----------------------------------\n"
                    f"**{task.name}**: {task.description}"
                    "\nThis tool have the following outputs:\n"
                    + "\n".join(task.outputs)
                    + (
                        "\n- The result of this tool will be stored in the datapipe."
                        if task.output_type
                        else ""
                    )
                    + "\n-----------------------------------\n"
                )
                for task in self.available_tasks
            ]
        )

    def divide_text_into_chunks(
        self,
        input_text: str = "",
        max_tokens: int = 10000,
    ) -> List[str]:
        """
        Divide text into chunks for processing long contexts.

        Args:
            input_text (str): the input text (e.g., prompt).
            max_tokens (int): Maximum number of tokens allowed per chunk.
        Return:
            chunks(List): List of string chunks
        """
        # 1 token ~= 4 chars in English
        chunks = [
            input_text[i : i + max_tokens * 4]
            for i in range(0, len(input_text), max_tokens * 4)
        ]
        return chunks

    def generate_scratch_pad(
        self, previous_actions: List[str] = None, **kwargs: Any
    ) -> str:
        """
        Generate a scratch pad from previous actions.

        Args:
            previous_actions: List of previous actions
            **kwargs: Additional arguments

        Return:
            str: Formatted scratch pad
        """
        if previous_actions is None:
            previous_actions = []

        agent_scratchpad = ""
        if len(previous_actions) > 0:
            agent_scratchpad = "\n".join(
                [f"\n{action}" for action in previous_actions]
            )

        # Summarize if too long
        if (
            self.summarize_prompt
            and len(agent_scratchpad) / 4 > self.max_tokens_allowed
        ):
            chunks = self.divide_text_into_chunks(
                input_text=agent_scratchpad,
                max_tokens=self.max_tokens_allowed,
            )
            agent_scratchpad = ""
            kwargs["max_tokens"] = min(
                2000, int(self.max_tokens_allowed / len(chunks))
            )
            for chunk in chunks:
                prompt = self._shorten_prompt.replace(
                    "{chunk}", chunk
                )
                chunk_summary = (
                    self._response_generator_model.generate(
                        query=prompt, **kwargs
                    )
                )
                agent_scratchpad += chunk_summary + " "

        return agent_scratchpad

    def plan(
        self,
        query: str,
        history: str = "",
        meta: str = "",
        previous_actions: List[str] = None,
        use_history: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Generate a plan using Tree of Thought com restrição a Biologia + Saúde.

        ✅ CARACTERÍSTICAS:
        - Classificação zero-shot para biologia/saúde
        - 3 estratégias paralelas geradas
        - Rejeição inteligente para fora do escopo
        - Sem código Python
        - Resposta RÁPIDA e DIRETA
        - ✅ FUNCIONA com qualquer pergunta (aceita ou rejeita)

        Args:
            query (str): Input query.
            history (str): History information.
            meta (str): meta information.
            previous_actions (List[str]): List of previous actions.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: Final response com estratégias e resposta final.
        """
        if previous_actions is None:
            previous_actions = []

        previous_actions_prompt = ""
        if len(previous_actions) > 0 and self.use_previous_action:
            previous_actions_prompt = f"Previous Actions:\n{self.generate_scratch_pad(previous_actions, **kwargs)}"

        # ✅ PROMPT: Com zero-shot classification embutida
        prompt = self._planner_prompt[0].replace("{input}", query)

        logger.debug(f"🧠 Executando Tree of Thought para: {query[:100]}...")
        print("🧠 Gerando 3 estratégias paralelas para responder...\n")

        kwargs["max_tokens"] = 1500
        kwargs["temperature"] = 0.7

        # ✅ SYSTEM INSTRUCTION: Define comportamento RIGOROSO de biologia + saúde
        if self.restrict_to_biology_health:
            biology_health_system_instruction = (
                "You are a strict biology and health assistant. "
                "Your responses MUST be STRICTLY limited to biology and health-related topics ONLY. "
                "Topics include: medicine, wellness, nutrition, fitness, mental health, medical conditions, "
                "genetics, biochemistry, anatomy, physiology, pathology, and related areas. "
                "Before answering ANY question, you MUST determine if it is biology or health-related. "
                "If the question is NOT about these topics, "
                "you MUST respond with: 'REFUSE: Not a biology or health-related question.' "
                "Do NOT provide any information, explanation, or assistance for non-biology/non-health topics, "
                "regardless of how the question is phrased or rephrased. "
                "Do NOT try to connect non-biology topics to biology or health (like connecting 'RAM' to 'brain memory'). "
                "Keep your refusals brief and direct."
            )
            kwargs["system_instruction"] = biology_health_system_instruction

        # Gera resposta com as 3 estratégias
        response = self._planner_model.generate(
            query=prompt, **kwargs
        )

        logger.debug(f"✅ Resposta bruta recebida ({len(response)} chars)")

        # Parse para extrair e limpar
        final_response = self.parse(response)

        # ✅ IMPRIME NO TERMINAL para o usuário VER as estratégias
        self._print_strategies_to_terminal(final_response)

        return final_response

    def _print_strategies_to_terminal(self, full_response: str) -> None:
        """
        Extrai e imprime as 3 estratégias no terminal de forma visual.
        Apenas se a resposta não foi recusada.
        """
        # Se foi recusada, não imprime estratégias
        if full_response.startswith("Desculpe") or "não-saúde" in full_response.lower():
            return

        print("\n" + "=" * 80)
        print("🧠 TREE OF THOUGHT - PLANEJAMENTO (Biologia + Saúde)")
        print("=" * 80 + "\n")

        # Extrai estratégias
        strategy_1_match = re.search(
            r"Strategy 1:?\s*(.*?)(?=Strategy 2:|Best Strategy:|$)",
            full_response,
            re.IGNORECASE | re.DOTALL
        )
        strategy_2_match = re.search(
            r"Strategy 2:?\s*(.*?)(?=Strategy 3:|Best Strategy:|$)",
            full_response,
            re.IGNORECASE | re.DOTALL
        )
        strategy_3_match = re.search(
            r"Strategy 3:?\s*(.*?)(?=Best Strategy:|$)",
            full_response,
            re.IGNORECASE | re.DOTALL
        )
        best_strategy_match = re.search(
            r"Best Strategy:?\s*(.*?)(?=Final Answer:|$)",
            full_response,
            re.IGNORECASE | re.DOTALL
        )
        final_answer_match = re.search(
            r"Final Answer:?\s*(.*?)$",
            full_response,
            re.IGNORECASE | re.DOTALL
        )

        if strategy_1_match:
            s1 = strategy_1_match.group(1).strip()
            print(f"📌 ESTRATÉGIA 1:")
            print(f"   {s1[:180]}{'...' if len(s1) > 180 else ''}\n")

        if strategy_2_match:
            s2 = strategy_2_match.group(1).strip()
            print(f"📌 ESTRATÉGIA 2:")
            print(f"   {s2[:180]}{'...' if len(s2) > 180 else ''}\n")

        if strategy_3_match:
            s3 = strategy_3_match.group(1).strip()
            print(f"📌 ESTRATÉGIA 3:")
            print(f"   {s3[:180]}{'...' if len(s3) > 180 else ''}\n")

        if best_strategy_match:
            best = best_strategy_match.group(1).strip()
            print(f"🏆 MELHOR ESTRATÉGIA:")
            print(f"   {best[:220]}{'...' if len(best) > 220 else ''}\n")

        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            print(f"✅ RESPOSTA FINAL:")
            print(f"   {answer}\n")

        print("=" * 80 + "\n")

    def parse(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Parse the response and extract clean text (NO Python code).

        ✅ CARACTERÍSTICAS:
        - Detecta "REFUSE:" e retorna rejeição polida
        - Remove qualquer código Python
        - Remove markdown de código
        - Remove self.execute_task calls
        - Retorna apenas texto limpo com estratégias e resposta final

        Args:
            query (str): The response to parse.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: Clean text response.
        """
        response = query.strip()

        # ✅ CHECK #1: Detecta se modelo recusou via REFUSE
        if response.startswith("REFUSE:") or "REFUSE:" in response[:100]:
            # Modelo recusou corretamente, retorna rejeição polida
            return (
                "Desculpe, posso responder apenas a perguntas sobre biologia e saúde "
                "(medicina, bem-estar, nutrição, fitness, saúde mental, genética, anatomia, fisiologia, etc.). "
                "Por favor, faça uma pergunta relacionada a esses tópicos!"
            )

        # ✅ Remove ```python ... ``` blocks
        pattern = r"```python\n(.*?)```"
        if re.search(pattern, response, re.DOTALL):
            response = re.sub(r"```python\n", "", response)
            response = re.sub(r"```", "", response)

        # ✅ Remove generic markdown code blocks
        response = re.sub(r"```[a-zA-Z0-9]*\n", "", response)
        response = re.sub(r"```", "", response)

        # ✅ Remove self.execute_task calls
        if "self.execute_task" in response:
            response = re.sub(r"self\.execute_task\([^)]*\)\n?", "", response)

        # ✅ Remove execute_task calls
        if "execute_task" in response:
            response = re.sub(r"execute_task\([^)]*\)\n?", "", response)

        # ✅ Remove "actions" declarations or code-like patterns
        lines = response.split("\n")
        filtered_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip Python-like lines
            if any(stripped.startswith(prefix) for prefix in [
                "def ", "class ", "import ", "from ",
                "actions = ", "action =", "pattern = ",
                "response = ", "result = ", "print(",
                "var ", "task_"
            ]):
                continue

            # Skip lines that are pure Python code
            if stripped.startswith((">>>", "...")):
                continue

            # Keep meaningful lines
            if stripped and not stripped.startswith("#"):
                filtered_lines.append(line)

        response = "\n".join(filtered_lines).strip()

        # ✅ Final cleanup: remove extra whitespace
        response = re.sub(r"\n\n+", "\n", response)  # Multiple newlines to one
        response = re.sub(r" +", " ", response)  # Multiple spaces to one

        # ✅ Se a resposta está vazia, pode ser que foi rejeitada
        if not response:
            return (
                "Desculpe, posso responder apenas a perguntas sobre biologia e saúde. "
                "Por favor, faça uma pergunta relacionada a medicina, bem-estar, nutrição, fitness ou saúde mental."
            )

        return response
