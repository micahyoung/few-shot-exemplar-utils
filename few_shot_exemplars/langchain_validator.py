from __future__ import annotations

from typing import List, Optional, Sequence, Dict, Any

from langchain.prompts import FewShotPromptTemplate, PromptTemplate


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
        self.examples = examples or prompt.examples

    def replay_test(self) -> str:
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

    def ablation_test(self) -> str:
        """Test the impact of each example by removing it and comparing answers.
        
        For each example, creates a prompt with that example removed and tests
        if the LLM still gives the same answer for that example's question.
        
        Returns:
            A diff-style string showing examples where removing the example
            changed the answer. If all examples are consistent, returns empty string.
        """
        example_question_key = "question"
        example_answer_key = "answer"
        prompt_input_key = "input"
        prompt_prefix = self.prompt.example_prompt.template.split(f"{{{example_question_key}}}")[1].strip().split(f"{{{example_answer_key}}}")[0]
        
        diffs: List[str] = []
        for i, example in enumerate(self.examples):
            # Create a new examples list without the current example
            ablated_examples = [ex for j, ex in enumerate(self.examples) if j != i]
            
            # Create a new prompt with the ablated examples
            ablated_prompt = FewShotPromptTemplate(
                examples=ablated_examples,
                example_prompt=self.prompt.example_prompt,
                prefix=self.prompt.prefix,
                suffix=self.prompt.suffix,
                input_variables=self.prompt.input_variables,
            )
            
            question = example[example_question_key]
            original_answer = str(example[example_answer_key]).strip()
            
            # Test with the ablated prompt
            chain = ablated_prompt | self.llm
            ablated_answer = chain.invoke({prompt_input_key: question}).content.strip().replace(prompt_prefix, "")
            
            if ablated_answer != original_answer:
                diff = "\n".join(
                    [
                        f"# Q: {question}",
                        f"- {original_answer}",
                        f"+ {ablated_answer}",
                    ]
                )
                diffs.append(diff)
        
        return "\n\n".join(diffs)


__all__ = ["ExemplarValidator"]
