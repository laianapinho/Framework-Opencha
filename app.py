from openCHA import openCHA
from openCHA.llms import LLMType

cha = openCHA(
	verbose = True,
	planner_llm = LLMType.GEMINI,
	response_generator_llm = LLMType.GEMINI
)
cha.run_with_interface()