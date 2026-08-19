# Generation - OpenRouter integration

## What it is

The API call layer that sends the assembled prompt (system + context +
question - see RAG.md for how that's built) to an LLM and returns its
answer. This file covers the OpenRouter/SDK-specific integration details;
the prompt-design decisions live in RAG.md.

## Connecting to OpenRouter

OpenRouter exposes an OpenAI-compatible API - the official `openai`
Python SDK works against it by just pointing `base_url` elsewhere:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # note: no /chat/completions here, the SDK appends the path itself
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages
)

answer = response.choices[0].message.content
```

## Decisions made in this project

- **Used `openrouter/free` instead of a hardcoded specific free model**:
  OpenRouter's free-tier catalog changes frequently and without notice -
  an initially-used model (`meta-llama/llama-3.1-8b-instruct:free`) was
  deprecated within the same project. `openrouter/free` is a router
  alias that picks among currently-available free models automatically.
  Trade-off accepted knowingly: loses control over which specific model
  answers a given query (relevant for reproducible evaluation later),
  but removes a real, recurring maintenance burden for a project with
  zero-cost constraints and no need for reproducibility during
  development.

## Problems encountered

- **A hardcoded free model became unavailable mid-project**: confirmed
  via search that OpenRouter had recently deprecated the entire free
  Meta Llama tier, breaking the originally hardcoded model. Root cause
  of the switch to `openrouter/free` above.