import os
from few_shot_exemplars.langchain_validator import ExemplarValidator
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

example_prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")

examples = [
    {
        "question": "Who died younger, Muhammad Ali or Alan Turing?",
        "answer": "Alan Turing 🏴󠁧󠁢󠁥󠁮󠁧󠁿: 41 years old",
    },
    {
        "question": "Who outlived whom, Robin or Maurice Gibb?",
        "answer": "Robin Gibb 🇬🇧: 62 years old",
    },
    {
        "question": "Who lived longer, Tina Turner or Marianne Faithfull?",
        "answer": "Tina Turner 🇺🇸: 100 years old", # wrong age for demo
    },
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Return your best guess, name/flag/age, without explanation",
    suffix="Q: {input}",
    input_variables=["input"],
)

llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    temperature=0.0,
    extra_body={"reasoning_effort": "minimal"} if os.environ["OPENAI_MODEL"].startswith("gpt-5") else {},
)

validator = ExemplarValidator(prompt, llm)

replay_result = validator.replay_test()
print("Replay test results:")
print(replay_result)

ablation_result = validator.ablation_test()
print("Ablation test results:")
print(ablation_result)
