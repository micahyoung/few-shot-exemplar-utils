import pytest
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from few_shot_exemplars.langchain_exemplars import ExemplarValidator


def test_exemplar_validator_detects_inconsistency():
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

    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    def mock_llm_func(prompt_dict):
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        formatted_prompt = prompt_dict["input"] if isinstance(prompt_dict, dict) else str(prompt_dict)
        if "Tina Turner or Ruby Turner" in formatted_prompt:
            return MockResponse("Tina Turner (83) \U0001F1FA\U0001F1F8")
        else:
            return MockResponse("Muhammad Ali (74) \U0001F1FA\U0001F1F8")

    llm = mock_llm_func
    validator = ExemplarValidator(examples, prompt, llm)

    diff = validator.replay_consistency()
    assert "Tina Turner (83)" in diff
    assert "- Ruby Turner (65)" in diff

    formatted = prompt.format(input="Who outlived who: Robin or Maurice Gibb?")
    assert formatted.strip().endswith(
        "Question: Who outlived who: Robin or Maurice Gibb?"
    )
