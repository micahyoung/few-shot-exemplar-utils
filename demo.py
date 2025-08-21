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
        "answer": "Tina Turner 🇺🇸: 100 years old",  # wrong age for demo
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

ablation_result = validator.ablation_test()
print("Ablation test results:")
print(ablation_result)
print("\n")

replay_result = validator.replay_test()
print("Replay test results (original examples):")
print(replay_result)
print("\n")

# Update examples with replay results
updated_prompt = prompt.model_copy()
updated_prompt.examples = validator.replay_examples()
updated_validator = ExemplarValidator(updated_prompt, llm)

updated_replay_result = updated_validator.replay_test()
print("Replay test results (updated examples):")
print(updated_replay_result)
