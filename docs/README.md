# Few-Shot Prompt Exemplar Utilities

Few-shot LLM prompts using curated exemplars is a powerful prompt-engineering technique to elicit consistent, desirable behavior from LLMs, allowing users to define style, substance, intent, knowledge, and boundaries using only intuitive examples and counter-examples. The structure of few-shot examples strongly aligns with labeled LLM training data, giving powerful post-training control without fine-tuning. Outside LLM context, exemplars serve as ground-truth, integration testing hooks, RAG retrieval context, and potentially training data.

The goal of these tools are to simplify operationalizing these techniques with common libraries and use-cases.

## Installation

```bash
pip install few-shot-exemplar-utils
```

## Langchain `ExemplarValidator`

### Usage

```python
from few_shot_exemplars.langchain_validator import ExemplarValidator
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

example_prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")

examples = [
    {
        "question": "Who died younger, Muhammad Ali or Alan Turing?",
        "answer": "Alan Turing 🏴󠁧󠁢󠁥󠁮󠁧󠁿: 41 years old",
    },
    {
        "question": "Who lived longer, Tina Turner or Ruby Turner?",
        "answer": "Tina Turner 🇺🇸: 100 years old",
    }
]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Return your best guess, without explanation",
    suffix="Q: {input}",
    input_variables=["input"],
)

llm = ChatOpenAI()

validator = ExemplarValidator(examples, prompt, llm)

result = validator.replay_consistency()
print(result)
```

Output
```diff
# Q: Who lived longer, Tina Turner or Ruby Turner?
- Tina Turner 🇺🇸: 100 years old
+ Tina Turner 🇺🇸: 83 years old
```
