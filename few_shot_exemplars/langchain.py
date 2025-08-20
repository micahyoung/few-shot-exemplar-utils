from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Dict, Any, Optional

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import os


class FewShotPromptTemplateBuilder:
    """Helper for building and validating LangChain few-shot prompts."""

    def __init__(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        example_prompt: PromptTemplate,
        suffix: str,
        input_variables: Sequence[str],
        prefix: str | None = "",
        llm: Optional[Any] = None,
    ) -> None:
        self.examples = list(examples)
        self.example_prompt = example_prompt
        self.suffix = suffix
        self.input_variables = list(input_variables)
        self.prefix = prefix or ""
        self._llm = llm

    @property
    def prompt(self) -> FewShotPromptTemplate:
        """Return the underlying :class:`~langchain.prompts.FewShotPromptTemplate`."""
        return FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            suffix=self.suffix,
            prefix=self.prefix,
            input_variables=self.input_variables,
        )

    def _call_llm(self, llm: Any, prompt: str) -> str:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            # Some LangChain LLMs return a message or string.
            if hasattr(result, "content"):
                return result.content
            return str(result)
        if hasattr(llm, "predict"):
            return llm.predict(prompt)
        if callable(llm):
            return llm(prompt)
        raise ValueError("Unsupported LLM interface")

    def replay_consistency(self, *, llm: Optional[Any] = None, num: int = 1) -> str:
        """Replay examples and show diffs for inconsistent answers.

        Args:
            llm: LLM to use. If ``None`` an ``OpenAI`` instance is created using
                the ``OPENAI_API_KEY`` environment variable.
            num: Unused placeholder for API compatibility. The examples are
                replayed once.

        Returns:
            A diff-style string highlighting mismatched answers. If all examples
            match, an empty string is returned.
        """

        llm_to_use = llm or self._llm
        if llm_to_use is None:
            from langchain.llms import OpenAI

            llm_to_use = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        diffs: List[str] = []
        for idx, example in enumerate(self.examples):
            others = self.examples[:idx] + self.examples[idx + 1 :]
            prompt = FewShotPromptTemplate(
                examples=others,
                example_prompt=self.example_prompt,
                suffix=self.suffix,
                prefix=self.prefix,
                input_variables=self.input_variables,
            )
            rendered = prompt.format(input=example.get("question"))
            actual = self._call_llm(llm_to_use, rendered).strip()
            expected = str(example.get("answer", "")).strip()
            if actual != expected:
                diff = "\n".join(
                    [
                        f"# Q: {example.get('question')}",
                        f"- {expected}",
                        f"+ {actual}",
                    ]
                )
                diffs.append(diff)
        return "\n\n".join(diffs)

__all__ = ["FewShotPromptTemplateBuilder"]
