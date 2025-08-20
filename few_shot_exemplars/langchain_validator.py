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

    def _get_example_keys(self) -> tuple[str, str]:
        """Extract question and answer keys from the examples."""
        if not self.examples:
            raise ValueError("No examples available to extract keys from")
        
        first_example = self.examples[0]
        keys = list(first_example.keys())
        return keys[0], keys[1]
    
    def _get_prompt_prefix(self) -> str:
        """Extract the prompt prefix for cleaning LLM responses."""
        example_question_key, example_answer_key = self._get_example_keys()
        return self.prompt.example_prompt.template.split(f"{{{example_question_key}}}")[1].strip().split(f"{{{example_answer_key}}}")[0]
    
    def _invoke_llm(self, prompt: FewShotPromptTemplate, question: str) -> str:
        """Invoke the LLM with a given prompt and question, returning cleaned response."""
        prompt_input_key = prompt.input_variables[0]
        prompt_prefix = self._get_prompt_prefix()
        chain = prompt | self.llm
        return chain.invoke({prompt_input_key: question}).content.strip().replace(prompt_prefix, "")
    
    def _create_diff(self, question: str, expected: str, actual: str) -> str:
        """Create a diff string for answers."""
        if expected == actual:
            return "\n".join([
                f"# Q: {question}", 
                "# (identical)"
            ])

        return "\n".join([
            f"# Q: {question}",
            f"- {expected}",
            f"+ {actual}",
        ])

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
            ablated_answer = self._invoke_llm(ablated_prompt, question)
            
            diff = self._create_diff(question, original_answer, ablated_answer)
            diffs.append(diff)
        
        return "\n\n".join(diffs)


__all__ = ["ExemplarValidator"]
