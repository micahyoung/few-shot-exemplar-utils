import pytest
from langchain.prompts import PromptTemplate

from few_shot_exemplars.langchain import FewShotPromptTemplateBuilder


def test_replay_consistency_detects_diff():
    examples = [
        {
            "question": "Who lived longer, Muhammad Ali or Alan Turing?",
            "answer": "Muhammad Ali (74) \U0001F1FA\U0001F1F8",
        },
        {
            "question": "Who lived longer, Tina Turner or Ruby Turner?",
            "answer": "Ruby Turner (65) \U0001F1EF\U0001F1F2",
        },
    ]
    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
    builder = FewShotPromptTemplateBuilder(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    class StubLLM:
        def __call__(self, prompt: str) -> str:
            question = prompt.strip().splitlines()[-1].replace("Question: ", "")
            answers = {
                "Who lived longer, Muhammad Ali or Alan Turing?": "Muhammad Ali (74) \U0001F1FA\U0001F1F8",
                "Who lived longer, Tina Turner or Ruby Turner?": "Tina Turner (83) \U0001F1FA\U0001F1F8",
            }
            return answers[question]

    diff = builder.replay_consistency(llm=StubLLM())
    assert "Tina Turner (83)" in diff
    assert "- Ruby Turner (65)" in diff

    prompt = builder.prompt
    formatted = prompt.format(input="Who outlived who: Robin or Maurice Gibb?")
    assert formatted.strip().endswith(
        "Question: Who outlived who: Robin or Maurice Gibb?"
    )
