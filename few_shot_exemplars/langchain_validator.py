from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain.prompts import FewShotPromptTemplate


class ExemplarValidator:
    """Validates examples for consistency using an LLM."""

    def __init__(
        self,
        prompt: FewShotPromptTemplate,
        llm: Any,
        examples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self.prompt = prompt
        self.llm = llm
        self.examples = examples or prompt.examples or []

    def _validate_examples(self) -> None:
        """Validate that examples are available."""
        if not self.examples:
            raise ValueError("No examples available")

    def _get_example_keys(self) -> tuple[str, str]:
        """Extract question and answer keys from the examples."""
        self._validate_examples()
        first_example = self.examples[0]
        keys = list(first_example.keys())
        return keys[0], keys[1]

    def _get_prompt_prefix(self) -> str:
        """Extract the prompt prefix for cleaning LLM responses."""
        example_question_key, example_answer_key = self._get_example_keys()
        return (
            self.prompt.example_prompt.template.split(f"{{{example_question_key}}}")[1]
            .strip()
            .split(f"{{{example_answer_key}}}")[0]
        )

    def _create_ablated_prompt(self, exclude_index: int) -> FewShotPromptTemplate:
        """Create a prompt with the example at exclude_index removed."""
        self._validate_examples()
        ablated_examples = [
            ex for j, ex in enumerate(self.examples) if j != exclude_index
        ]
        return FewShotPromptTemplate(
            examples=ablated_examples,
            example_prompt=self.prompt.example_prompt,
            prefix=self.prompt.prefix,
            suffix=self.prompt.suffix,
            input_variables=self.prompt.input_variables,
        )

    def _invoke_llm(self, prompt: FewShotPromptTemplate, question: str) -> str:
        """Invoke the LLM with a given prompt and question, returning cleaned response."""
        prompt_input_key = prompt.input_variables[0]
        prompt_prefix = self._get_prompt_prefix()
        chain = prompt | self.llm
        return (
            chain.invoke({prompt_input_key: question})
            .content.strip()
            .replace(prompt_prefix, "")
        )

    def _create_diff(self, question: str, expected: str, actual: str) -> str:
        """Create a diff string for answers."""
        if expected == actual:
            return "\n".join([f"# Q: {question}", "# (identical)"])

        return "\n".join(
            [
                f"# Q: {question}",
                f"- {expected}",
                f"+ {actual}",
            ]
        )

    def replay_test(self) -> str:
        """Replay examples and show diffs for all answers.

        Returns:
            A diff-style string showing all examples. Identical answers are marked
            as "(identical)", mismatched answers show the diff.
        """
        example_question_key, example_answer_key = self._get_example_keys()

        diffs: List[str] = []
        for example in self.examples:
            question = example[example_question_key]
            actual = self._invoke_llm(self.prompt, question)
            expected = str(example[example_answer_key]).strip()
            diff = self._create_diff(question, expected, actual)
            diffs.append(diff)
        return "\n\n".join(diffs)

    def ablation_test(self) -> str:
        """Test the impact of each example by removing it and comparing answers.

        For each example, creates a prompt with that example removed and tests
        if the LLM still gives the same answer for that example's question.

        Returns:
            A diff-style string showing all examples. Identical answers are marked
            as "(identical)", changed answers show the diff.
        """
        example_question_key, example_answer_key = self._get_example_keys()

        diffs: List[str] = []
        for i, example in enumerate(self.examples):
            ablated_prompt = self._create_ablated_prompt(i)
            question = example[example_question_key]
            original_answer = str(example[example_answer_key]).strip()
            ablated_answer = self._invoke_llm(ablated_prompt, question)
            diff = self._create_diff(question, original_answer, ablated_answer)
            diffs.append(diff)

        return "\n\n".join(diffs)

    def replay_examples(self) -> List[Dict[str, Any]]:
        """Return a new set of examples with answers replayed through the same mechanism as replay_test().

        Uses the same mechanism as replay_test() but returns the examples with
        LLM-generated answers instead of showing diffs.

        Returns:
            A list of examples with replayed answers from the LLM.
        """
        example_question_key, example_answer_key = self._get_example_keys()

        replayed_examples: List[Dict[str, Any]] = []
        for example in self.examples:
            question = example[example_question_key]
            replayed_answer = self._invoke_llm(self.prompt, question)

            # Create new example with replayed answer
            replayed_example = dict(example)
            replayed_example[example_answer_key] = replayed_answer
            replayed_examples.append(replayed_example)

        return replayed_examples

    def ablation_examples(self) -> List[Dict[str, Any]]:
        """Return a new set of examples with answers rewritten through the same mechanism as ablation_test().

        For each example, creates a prompt with that example removed and generates
        a new answer for that example's question using the ablated prompt.

        Returns:
            A list of examples with ablated answers from the LLM.
        """
        example_question_key, example_answer_key = self._get_example_keys()

        ablated_examples: List[Dict[str, Any]] = []
        for i, example in enumerate(self.examples):
            ablated_prompt = self._create_ablated_prompt(i)
            question = example[example_question_key]
            ablated_answer = self._invoke_llm(ablated_prompt, question)

            # Create new example with ablated answer
            ablated_example = dict(example)
            ablated_example[example_answer_key] = ablated_answer
            ablated_examples.append(ablated_example)

        return ablated_examples


__all__ = ["ExemplarValidator"]
