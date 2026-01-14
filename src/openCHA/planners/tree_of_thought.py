"""
Heavily borrowed from langchain: https://github.com/langchain-ai/langchain/

✅ FINAL - SEM KEYWORDS:
   - Zero-Shot Classification (o LLM decide se é saúde)
   - Sem lista infinita de keywords
   - Restrição de SAÚDE apenas via SYSTEM PROMPT + Zero-Shot
   - Permite que o modelo decida o que é saúde
   - Não gera código Python desnecessário
   - Planejamento MÍNIMO para respostas rápidas
   - ✅ FUNCIONA com "amor", "RAM", qualquer pergunta!
"""
import re
from typing import Any
from typing import List

from openCHA.planners import Action
from openCHA.planners import BasePlanner
from openCHA.planners import PlanFinish


class TreeOfThoughtPlanner(BasePlanner):
    """
    **Description:**

        This class implements Tree of Thought planner, which inherits from the BasePlanner base class.
        Tree of Thought employs parallel chain of thoughts startegies and decides which one is more
        suitable to proceed to get to the final answer.
        `Paper <https://arxiv.org/abs/2305.10601>`_

        ✅ FINAL - SOLUÇÃO SEM KEYWORDS:
           - Zero-Shot Classification (LLM decide)
           - Restrição APENAS via system prompt + zero-shot
           - O modelo decide inteligentemente o que é saúde
           - Não gera código Python desnecessário
           - Planejamento MÍNIMO = respostas RÁPIDAS
           - Evita chamar tasks inexistentes
           - ✅ FUNCIONA com perguntas ambíguas (amor, etc)
    """

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000
    restrict_to_health_only: bool = True  # ✅ ATIVO - Domínio de saúde

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "zero-shot-react-planner"

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
    def _shorten_prompt(self):
        return (
            "Summarize the following text. Make sure to keep the main ideas "
            "and objectives in the summary. Keep the links "
            "exactly as they are: "
            "{chunk}"
        )

    @property
    def _planner_prompt(self):
        """
        ✅ PROMPT COM ZERO-SHOT CLASSIFICATION:
        - Instrui o modelo a classificar se é saúde DENTRO do prompt
        - Se não é saúde, deve responder com "REFUSE:"
        - Se é saúde, responde normalmente
        - SEM keywords hardcoded
        - O LLM decide inteligentemente
        """
        return [
            """You are a helpful health and wellness assistant. Your role is STRICTLY limited to health-related topics.

IMPORTANT: First, determine if the question is about health, medicine, wellness, nutrition, fitness, mental health, or medical conditions.

If the question is NOT related to any of these health topics:
- Respond ONLY with: "REFUSE: Not a health-related question."
- Do NOT provide any information about non-health topics
- Do NOT explain why it's not health-related

If the question IS health-related:
- Provide a direct, helpful answer in plain language
- Do NOT generate Python code or function calls
- Do NOT generate tool descriptions or execute commands

Your response MUST be in the same language as the question.

Question: {input}

Answer:""",
        ]

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
        Generate a plan using Zero-Shot Classification approach.

        ✅ CARACTERÍSTICAS:
        - UM ÚNICO PROMPT com zero-shot classification
        - LLM decide se é saúde ou não
        - Sem lista de keywords
        - Restrição de domínio APENAS via system prompt + zero-shot
        - Planejamento MÍNIMO (direto para resposta)
        - Sem código Python
        - Resposta RÁPIDA
        - ✅ FUNCIONA com qualquer pergunta

        Args:
            query (str): Input query.
            history (str): History information.
            meta (str): meta information.
            previous_actions (List[str]): List of previous actions.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: Final response in plain text format.
        """
        if previous_actions is None:
            previous_actions = []

        previous_actions_prompt = ""
        if len(previous_actions) > 0 and self.use_previous_action:
            previous_actions_prompt = f"Previous Actions:\n{self.generate_scratch_pad(previous_actions, **kwargs)}"

        # ✅ ÚNICO PROMPT: Com zero-shot classification embutida
        prompt = (
            self._planner_prompt[0]
            .replace("{input}", query)
        )

        print("🧠 Health Domain Prompt (Zero-Shot Classification):\n", prompt)
        kwargs["max_tokens"] = 1000
        kwargs["temperature"] = 0.7

        # ✅ SYSTEM INSTRUCTION: Define comportamento RIGOROSO de saúde
        if self.restrict_to_health_only:
            health_system_instruction = (
                "You are a strict health and wellness assistant. "
                "Your responses MUST be STRICTLY limited to health-related topics ONLY. "
                "Before answering ANY question, you MUST determine if it is health-related. "
                "If the question is NOT about health, medicine, wellness, nutrition, fitness, mental health, or medical conditions, "
                "you MUST respond with: 'REFUSE: Not a health-related question.' "
                "Do NOT provide any information, explanation, or assistance for non-health topics, "
                "regardless of how the question is phrased or rephrased. "
                "Do NOT try to connect non-health topics to health (like connecting 'RAM' to 'memory' and brain health). "
                "Keep your refusals brief and direct."
            )
            kwargs["system_instruction"] = health_system_instruction

        response = self._planner_model.generate(
            query=prompt, **kwargs
        )

        print("✅ Response:\n", response)

        # ✅ PARSE: Extract clean text response
        final_response = self.parse(response)

        return final_response

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
        - Retorna apenas texto limpo

        Args:
            query (str): The response to parse.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: Clean text response.
        """
        response = query.strip()

        # ✅ CHECK #1: Detecta se modelo recusou via REFUSE
        if response.startswith("REFUSE:") or "REFUSE:" in response[:50]:
            # Modelo recusou corretamente, retorna rejeição polida
            return (
                "Desculpe, posso responder apenas a perguntas sobre saúde, medicina, "
                "bem-estar, nutrição, fitness e saúde mental. "
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

        # ✅ Se a resposta está vazia, pode ser que foi rejeitada por não ser saúde
        if not response:
            return (
                "Desculpe, posso responder apenas a perguntas sobre saúde e bem-estar. "
                "Por favor, faça uma pergunta sobre saúde, medicina, nutrição, fitness ou saúde mental."
            )

        return response
