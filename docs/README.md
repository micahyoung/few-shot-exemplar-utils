# Few-Shot Prompt Exemplar Utilities

Few-shot LLM prompts using curated exemplars is a powerful prompt-engineering technique to elicit consistent, desirable behavior from LLMs, allowing users to define style, substance, intent, knowledge, and boundaries using only intuitive examples and counter-examples. The structure of few-shot examples strongly aligns with labeled LLM training data, giving powerful post-training control without fine-tuning. Outside LLM context, exemplars serve as ground-truth, integration testing hooks, RAG retrieval context, and potentially training data.

The goal of these tools are to simplify operationalizing these techniques with common libraries and use-cases.

## Langchain `FewShotPromptTemplateBuilder`

### Usage

```python
from few_shot_exemplars.langchain import FewShotPromptTemplateBuilder


example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

[
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": "Muhammad Ali 🇺🇸",
    },
    {
        "question": "Who died younger, Tina Turner or Ruby Turner?",
        "answer": "Ruby Turner 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    }
]

prompt_builder = FewShotPromptTemplateBuilder(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

results = prompt_builder.replay_test(num=3)
print(results)
```

Output
```diff
```
