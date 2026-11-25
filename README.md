# Full-Stack AI, LLM & Agentic Engineering

Code for the **Full-Stack AI, LLM & Agentic Engineering** video course by WhiskerBrief. The course shows how to build a complete AI application from start to finish: one that understands questions, looks up the right information, uses tools, takes actions and keeps working for real users. It starts with the Python skills every AI project needs, then follows one guided project, **CareDesk**, an assistant for a fictional hospital, built only on invented data (nothing here is medical advice).

Each video has one folder, `part-NN-<topic>/`, with:

- `README.md` - the learning goal, a map of the files, exact commands, expected output and honest notes;
- `code/` - the runnable scripts and tests, with plain-language headers, step-by-step comments and docstrings;
- `data/` - small synthetic inputs and the real saved outputs of the recorded run.

Only code lives here: no slides, animations, images or videos.

## Parts

| # | Video | Code | Article |
|---|---|---|---|
| 1 | [Python engineering for AI](https://youtu.be/lRv88z2V2A8) | [part-01-python-engineering-for-ai](part-01-python-engineering-for-ai) | [Medium article](https://arjun-sarkar786.medium.com/1-full-stack-ai-llm-agentic-engineering-python-engineering-for-ai-662ed14f7f79) |
| 2 | Deep learning refresher with PyTorch | [part-02-deep-learning-refresher](part-02-deep-learning-refresher) | Medium article: link added when published |
| 3 | Language modelling basics | [part-03-language-modelling-basics](part-03-language-modelling-basics) | Medium article: link added when published |
| 4 | Tokenization | [part-04-tokenization](part-04-tokenization) | Medium article: link added when published |

More folders are added as each video is released.

## How the code is run

Everything runs for free. Code runs on a laptop first, then on free Google Colab where a GPU helps. Steps that need rented GPUs or paid services are shown as code and are marked in their README ("shown, not run") so anyone with a capable GPU or a cloud account can run them. Secrets never go into this repository: copy `.env.example` to `.env` and keep your own keys there.

## Prerequisites

Basic coding skills and a little familiarity with machine learning. Every important idea is introduced before it is used.
