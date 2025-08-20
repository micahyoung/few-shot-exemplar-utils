from __future__ import annotations

from typing import List, Sequence, Dict, Any

from langchain.prompts import FewShotPromptTemplate, PromptTemplate


class ExemplarValidator:
    """Validates examples for consistency using an LLM."""
    
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        prompt: FewShotPromptTemplate,
        llm: Any,
    ) -> None:
        self.examples = list(examples)
        self.prompt = prompt
        self.llm = llm

    def replay_consistency(self) -> str:
        """Replay examples and show diffs for inconsistent answers.

        Returns:
            A diff-style string highlighting mismatched answers. If all examples
            match, an empty string is returned.
        """
        example_question_key = "question"
        example_answer_key = "answer"
        prompt_input_key = "input"
        prompt_prefix = self.prompt.example_prompt.template.split(f"{{{example_question_key}}}")[1].strip().split(f"{{{example_answer_key}}}")[0]

        diffs: List[str] = []
        for example in self.examples:
            question = example[example_question_key]
            chain = self.prompt | self.llm
            actual = chain.invoke({prompt_input_key: question}).content.strip().replace(prompt_prefix, "")
            expected = str(example[example_answer_key]).strip()
            if actual != expected:
                diff = "\n".join(
                    [
                        f"# Q: {question}",
                        f"- {expected}",
                        f"+ {actual}",
                    ]
                )
                diffs.append(diff)
        return "\n\n".join(diffs)


__all__ = ["ExemplarValidator"]
