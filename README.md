## Generative Agents

<hr>

*Generative agents codebase.*

<br>

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

`generative-agents` is a package that contains:

- Agent classes
- Workflow scripts

## Documentation

The official documentation is hosted on this repository's [wiki](), along with a long-term roadmap. A short-term todo list for this project is posted on our [kanban board]().


## Installation

### Linux

Set up a Python 3.11.3 virtual environment, then make the following local invocations from the terminal:

```
pip install -e .[linux]

pre-commit install

pre-commit autoupdate
```

### Mac

Set up a Python 3.11 conda environment by making the following local invocations from the terminal:

```
conda create --name generative_agents_env python=3.11

conda activate generative_agents_env

pip install -e .[mac]

pre-commit install

pre-commit autoupdate
```

## Unit tests

After installation, make the following local invocation from the terminal:
```
pytest
```

## Quick Start

Defining your API keys with the following environment variable:
```
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

Running a conversation locally:
```
python workflows/agent/model/conversation.py --llmType=GPT-4o --temperature=0.9
```

Running the penpal app locally:
```
python workflows/agent/model/penpal.py --llmType=GPT-4o --temperature=0.9

streamlit run penpal_app/app.py
```

Running an interview locally, first add your data file to ~/datasets, then:
```
python workflows/agent/model/interview.py --llmType=Claude-v1 --temperature=0.9
```
