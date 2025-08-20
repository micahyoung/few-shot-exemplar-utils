# Few-Shot Prompt Exemplar Utilities

Few-shot LLM prompts using curated exemplars is a powerful prompt-engineering technique to elicit consistent, desirable behavior from LLMs, allowing users to define style, substance, intent, knowledge, and boundaries using only intuitive examples and counter-examples. The structure of few-shot examples strongly aligns with labeled LLM training data, giving powerful post-training control without fine-tuning. Outside LLM context, exemplars serve as ground-truth, integration testing hooks, RAG retrieval context, and potentially training data.

The goal of these tools are to simplify operationalizing these techniques with common libraries and use-cases.

## Installation

```bash
pip install few-shot-exemplar-utils
```

## Langchain `FewShotPromptTemplateBuilder`

### Requirements
- `OPENAI_API_KEY` key (or pass an `llm` option to `replay_consistency`)

### Usage

```python
from few_shot_exemplars.langchain import FewShotPromptTemplateBuilder
from langchain.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": "Muhammad Ali (74) 🇺🇸",
    },
    {
        "question": "Who lived longer, Tina Turner or Ruby Turner?",
        "answer": "Ruby Turner (65) 🇯🇲",
    }
]

prompt_builder = FewShotPromptTemplateBuilder(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

result = prompt_builder.replay_consistency()
print(result)
```

Output
```diff
# Q: Who lived longer, Tina Turner or Ruby Turner?
- Ruby Turner (65) 🇯🇲
+ Tina Turner (83) 🇺🇸
```

Then use the prompt normally:
```python
prompt = prompt_builder.prompt

print(
    prompt.invoke({"input": "Who outlived who: Robin or Maurice Gibb?"}).to_string()
)
```

Output
```text
Robin Gibb (62) 🇬🇧
```
