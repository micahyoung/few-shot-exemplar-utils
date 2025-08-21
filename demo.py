import os

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

from few_shot_exemplars.langchain_validator import ExemplarValidator

example_prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")

examples = [
    {
        "question": "Who died younger, Alan Turing or Muhammad Ali?",
        "answer": "Alan Turing 🏴󠁧󠁢󠁥󠁮󠁧󠁿: 41 years old",
    },
    {
        "question": "Who outlived whom, Maurice Gibb or Robin Gibb?",
        "answer": "Robin Gibb 🇬🇧: 62 years old",
    },
    {
        "question": "Who lived longer, Marianne Faithfull or Tina Turner?",
        "answer": "Tina Turner 🇺🇸: 100 years old",  # wrong - Tina died at 83
    },
    {
        "question": "Who died first, John Lennon or George Harrison?",
        "answer": "George Harrison 🇬🇧: 58 years old",  # wrong - Lennon died first
    },
    {
        "question": "Who was older at death, Elvis Presley or Michael Jackson?",
        "answer": "Elvis Presley 🇺🇸: 75 years old",  # wrong - Elvis died at 42, Michael at 50
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
    extra_body=(
        {"reasoning_effort": "minimal"}
        if os.environ["OPENAI_MODEL"].startswith("gpt-5")
        else {}
    ),
)

validator = ExemplarValidator(prompt, llm)

# Ablation method
ablation_result = validator.ablation_test()
print("Ablation test results (original examples):")
print(ablation_result)
print("\n")

ablated_prompt = prompt.model_copy()
ablated_prompt.examples = validator.ablation_examples()
ablated_validator = ExemplarValidator(ablated_prompt, llm)

updated_ablation_result = ablated_validator.ablation_test()
print("Ablation test results (ablated examples):")
print(updated_ablation_result)
print("\n")

# Replay method
replay_result = validator.replay_test()
print("Replay test results (original examples):")
print(replay_result)
print("\n")

replayed_prompt = prompt.model_copy()
replayed_prompt.examples = validator.replay_examples()
replayed_validator = ExemplarValidator(replayed_prompt, llm)

updated_replay_result = replayed_validator.replay_test()
print("Replay test results (replayed examples):")
print(updated_replay_result)
