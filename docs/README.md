# Few-Shot Prompt Exemplar Utilities

Few-shot LLM prompts using curated exemplars is a powerful prompt-engineering technique to elicit consistent, desirable behavior from LLMs, allowing users to define style, substance, intent, knowledge, and boundaries using only intuitive examples and counter-examples. The structure of few-shot examples strongly aligns with labeled LLM training data, giving powerful post-training control without fine-tuning. Outside LLM context, exemplars serve as ground-truth, integration testing hooks, RAG retrieval context, and potentially training data.

The goal of these tools are to simplify operationalizing these techniques with common libraries and use-cases.

## Installation

```bash
pip install few-shot-exemplar-utils
```

## MCP Server

This MCP [Model Context Protocol](https://modelcontextprotocol.io/llms.txt) implementation demonstrates direct curation and consumtion of exemplars through LLM-native interface, using [sampling](https://modelcontextprotocol.io/specification/2025-06-18/client/sampling) and [prompts](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts).

### Usage

1. Configure and activate the MCP via config:
    ```json
    {
        "servers": {
            "few-shot-exemplars": {
                "type": "stdio",
                "command": ".venv/bin/python",
                "args": [
                    "./few_shot_exemplars/mcp_server.py"
                ]
            }
        }
    }
    ```
1. Register your prompts and exemplars  with a Chat message like:

    ```text
    few-shot-exemplars:

    add a new prompt:
    > Return your best guess, name/flag/age, without explanation

    then, add each of these exemplars:
    Q: Who died younger, Muhammad Ali or Alan Turing?
    A: Alan Turing 🇬🇧: 41 years old

    Q: Who outlived whom, Robin or Maurice Gibb?
    A: Robin Gibb 🇬🇧: 62 years old

    Q: Who lived longer, Tina Turner or Marianne Faithfull?
    A: Tina Turner 🇺🇸: 100 years old
    ```

1. Correct any un-replayable questions:
    ```diff
    -Tina Turner 🇺🇸: 100 years old
    +Tina Turner 🇺🇸: 83 years old
    ```

1. Ask a novel question:
    ```text
    Q: Who lived longer, Louis Armstrong or Ella Fitzgerald?
    ```

1. (Optional) Load the prompt then ask a new question ([VS Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers#_use-mcp-prompts))

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
        "answer": "Tina Turner 🇺🇸: 100 years old", # wrong age
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

validator = ExemplarValidator(prompt, llm)
```

#### Replay test
```python

result = validator.replay_test()
print(result)
```

Output
```diff
# Q: Who died younger, Muhammad Ali or Alan Turing?
# (identical)

# Q: Who lived longer, Tina Turner or Ruby Turner?
- Tina Turner 🇺🇸: 100 years old
+ Tina Turner 🇺🇸: 83 years old
```

#### Ablation test
```python

result = validator.ablation_test()
print(result)
```

Output
```diff
# Q: Who died younger, Muhammad Ali or Alan Turing?
- Alan Turing 🏴󠁧󠁢󠁥󠁮󠁧󠁿: 41 years old
+ Alan Turing 🇬🇧: 41 years old

# Q: Who lived longer, Tina Turner or Ruby Turner?
- Tina Turner 🇺🇸: 100 years old
+ Tina Turner 🇺🇸: 83 years old
