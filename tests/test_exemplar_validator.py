import pytest
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from few_shot_exemplars.langchain_validator import ExemplarValidator


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
    validator = ExemplarValidator(prompt, llm)

    diff = validator.replay_test()
    assert "Tina Turner (83)" in diff
    assert "- Ruby Turner (65)" in diff

    formatted = prompt.format(input="Who outlived who: Robin or Maurice Gibb?")
    assert formatted.strip().endswith(
        "Question: Who outlived who: Robin or Maurice Gibb?"
    )


def test_ablation_test_detects_example_dependency():
    examples = [
        {
            "question": "Who lived longer, Muhammad Ali or Alan Turing?",
            "answer": "Muhammad Ali (74)",
        },
        {
            "question": "Who lived longer, Tina Turner or Ruby Turner?",
            "answer": "Ruby Turner (65)",
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
        
        # Simulate that without the Ruby Turner example, the model gives a different answer
        if "Tina Turner or Ruby Turner" in formatted_prompt:
            if "Ruby Turner (65)" not in str(formatted_prompt):
                # Without the Ruby Turner example, model gives wrong answer
                return MockResponse("Tina Turner (83)")
            else:
                # With the example, model gives correct answer
                return MockResponse("Ruby Turner (65)")
        else:
            return MockResponse("Muhammad Ali (74)")

    llm = mock_llm_func
    validator = ExemplarValidator(prompt, llm)

    diff = validator.ablation_test()
    assert "# Q: Who lived longer, Tina Turner or Ruby Turner?" in diff
    assert "- Ruby Turner (65)" in diff
    assert "+ Tina Turner (83)" in diff


def test_ablation_test_no_dependency():
    examples = [
        {
            "question": "Who lived longer, Muhammad Ali or Alan Turing?",
            "answer": "Muhammad Ali (74)",
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
        
        # Always returns the same answer regardless of examples
        return MockResponse("Muhammad Ali (74)")

    llm = mock_llm_func
    validator = ExemplarValidator(prompt, llm)

    diff = validator.ablation_test()
    assert diff == ""  # No differences, so empty string
