# AgentInspect

## Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis (ICLR 2026)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP/agent-quality-inspect)](https://api.reuse.software/info/github.com/SAP/agent-quality-inspect)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-red.svg)](https://iclr.cc/Conferences/2026)

Paper Link: https://openreview.net/pdf?id=fHsVNklKOc

Documentation Link: https://sap.github.io/agent-quality-inspect/

## Table of Contents
- [Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](#talk-evaluate-diagnose-user-aware-agent-evaluation-with-automated-error-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Quick Start](#quick-start)
    - [Option 1. Using it as a Metrics Package](#option-1-using-it-as-a-metrics-package)
    - [Option 2. Using it via the provided runners](#option-2-using-it-via-the-provided-runners)
    - [Viewing Results](#viewing-results)
    - [Error Diagnosis UI](#error-diagnosis-ui)
  - [Bring Your Own Agent](#bring-your-own-agent)
    - [Creating your own evaluation dataset](#creating-your-own-evaluation-dataset)
  - [Known Issues](#known-issues)
  - [How to obtain support](#how-to-obtain-support)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)

## Overview

![Two-step automated error discovery approach. Identical error colors indicate
that similar low-level errors are clustered into the same high-level category.](error_analysis_framework.png)


This repository contains the implementation of **Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis (TED)**.

The agent-quality-inspect toolkit evaluates agentic systems under different user personas (expert and non-expert), reports metrics such as **Area Under the Curve (AUC)**, **Progress Per Turn (PPT)**, **pass@k**, **pass^k**, etc., and provides detailed error analysis to identify specific areas for improvement in the agent.

At the core of TED is a **subgoal-based evaluation**: users specify a set of natural-language subgoals (e.g., "Agent should call `search_messages` after getting the current timestamp" or "Agent should explicitly state the final answer and justification") in the evaluation dataset. During evaluation, TED:

- Treats these subgoals (the `SubGoal` objects in the `EvaluationSample` schema) as the ground truth of what the agent should achieve over the course of the interaction.
- Compares each subgoal against the **agent trace** (`AgentDialogueTrace`), including turns, intermediate tool calls, and agent responses.
- Uses an LLM-as-a-judge to decide, for each turn, whether the behavior observed in the trace satisfies the relevant subgoals, and aggregates these judgments into progress curves and downstream metrics such as AUC and PPT.

### Features

- **User personas**: User proxy simulating users with different levels of domain expertise (expert and non-expert).
- **Metrics**:
  - **Area Under the Curve (AUC)**: Measures the area under the progress curve and evaluates how quickly the agent makes progress toward the goal.
  - **Progress Per Turn (PPT)**: Measures the average progress made by the agent in each turn.
  - **Task success and reliability**: Metrics such as pass@k and pass^k.
- **Error analysis**: Automatic categorization of failure modes to highlight where and why agents fail.
- **Agent-runner support**: Supports and is extensible to different agent runners. This repository currently includes integrations for **Tau2Bench** and **Toolsandbox** and provides a pattern to extend to other multi-turn benchmarks.

## Installation

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI API access (for evaluation metrics and the user proxy)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/SAP/agent-quality-inspect
cd agent-quality-inspect
```

2. Install the package in editable mode:

```bash
pip install -e .
```

3. Configure Azure OpenAI API credentials by creating a `.env` file in the project root:

```bash
AZURE_API_VERSION=your_api_version         # e.g., 2024-02-15-preview
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_KEY=your_api_key
```

4. (Optional) Set up agent runners:

- Tau2Bench runner setup: [agent_runners/README_tau2_bench_setup.md](agent_runners/README_tau2_bench_setup.md)
- Toolsandbox runner setup: [agent_runners/README_tool_sandbox_setup.md](agent_runners/README_tool_sandbox_setup.md)

> **Note:** The RapidAPI services used in these agent runners are third-party, user-subscribed services. SAP does not, provide, license or authorize their use.

## Quick Start

There are two primary ways to use this repository:

1. **As a metrics / evaluation package** inside your own code.
2. **Via the provided experiment runners** to reproduce paper results and run new evaluations.

### Option 1. Using it as a Metrics Package

The standard flow of using it as a metrics package is as follows:

1. To use the package as an importable dependency, enter your command terminal and use this command.

```bash
pip install git+https://github.com/SAP/agent-quality-inspect.git
```

2. Define your evaluation sample and agent trace.
3. Evaluate the progress rates using the evaluation sample on your agent trace.
4. Using the output of the Step 3, calculate any of the metric scores (AUC, PPT, pass@k, etc.).
5. Optionally, run the error analysis on the outputs of previous steps.
6. Visualize the error analysis results in the Streamlit UI.
   
Example: constructing a minimal trace and computing an AUC score with the metrics package: Run the script using `streamlit run <script>.py`.

```python
from typing import List
from agent_inspect.clients import AzureOpenAIClient
from agent_inspect.metrics.scorer import AUC, ProgressScoresThroughTurns
from agent_inspect.metrics.constants import (
    INCLUDE_VALIDATION_RESULTS,
    INCLUDE_JUDGE_EXPLANATION,
    OPTIMIZE_JUDGE_TRIALS
)
from agent_inspect.models.metrics.agent_trace import (
    AgentDialogueTrace,
    TurnTrace,
    AgentResponse,
)
from agent_inspect.models.metrics.agent_data_sample import EvaluationSample, SubGoal
from agent_inspect.models.tools import ErrorAnalysisDataSample
from agent_inspect.tools import ErrorAnalysis
from demo.ui_for_agent_diagnosis.app import launch_ui


# Create LLM client (requires env vars: AZURE_API_VERSION, AZURE_API_BASE, AZURE_API_KEY)
client = AzureOpenAIClient(model="gpt-4.1", max_tokens=4096)

# Build a minimal agent trace with a single turn
agent_trace = AgentDialogueTrace(
    turns=[
        TurnTrace(
            id="turn_1",
            agent_input="What is my current account balance?",
            agent_response=AgentResponse(
                response="Your current balance is 100 USD.",
            ),
        )
    ]
)

# 2. Define the evaluation data sample and subgoals
data_sample = EvaluationSample(
    sub_goals=[
        SubGoal(
            details="Agent should correctly state the user's current account balance.",
        )
    ]
)

# Step 1: Calculate progress rates using the evaluation sample and agent trace
progress_metric = ProgressScoresThroughTurns(
    llm_client=client,
    config={
        INCLUDE_VALIDATION_RESULTS: True,
        INCLUDE_JUDGE_EXPLANATION: True,
        OPTIMIZE_JUDGE_TRIALS: False
    }
)
progress_scores = progress_metric.evaluate(
    agent_trace=agent_trace,
    evaluation_data_sample=data_sample
)

print(f"Progress scores calculated for {len(progress_scores)} turn(s)")
for i, score in enumerate(progress_scores, 1):
    print(f"  Turn {i}: {score.score:.2f}")


# Step 2: Calculate AUC from progress scores
auc_result = AUC.get_auc_score_from_progress_scores(progress_scores)
print(f"AUC score: {auc_result.score:.2f}")


# Step 3: Run error analysis

# Extract validation results from the final turn
subgoal_validations = progress_scores[-1].validation_results

# Prepare data for error analysis
error_analysis_data_samples: List[ErrorAnalysisDataSample] = [
    ErrorAnalysisDataSample(
        data_sample_id=1,
        agent_run_id=1,
        subgoal_validations=subgoal_validations,
    )
]

# Run error analysis
error_analyzer = ErrorAnalysis(llm_client=client, max_workers=3)
error_analysis_result = error_analyzer.analyze_batch(error_analysis_data_samples)

# Display results
error_categories = list(error_analysis_result.analyzed_validations_clustered_by_errors.keys())
print(f"\nIdentified {len(error_categories)} error categories:")
for i, category in enumerate(error_categories, 1):
    count = len(error_analysis_result.analyzed_validations_clustered_by_errors[category])
    print(f"  {i}. {category} ({count} occurrences)")
print(f"Completed validations: {len(error_analysis_result.completed_subgoal_validations)}")


# Step 4: Launch UI for visualization
launch_ui(
    error_analysis_result=error_analysis_result,
    data_samples=error_analysis_data_samples
)
```

For more information on viewing error analysis UI [demo/ui_for_agent_diagnosis/readme.md](demo/ui_for_agent_diagnosis/readme.md).


### Option 2. Using it via the provided runners

Benchmarks are orchestrated via the runners in [paper_experiments](paper_experiments/readme.md) and external agent runners (for example, Tau2Bench or ToolSandbox).

1. **Start your agent runner** (for example, Tau2Bench):
  - Set up the Tau2Bench environment as described in [agent_runners/README_tau2_bench_setup.md](agent_runners/README_tau2_bench_setup.md).

2. **Run the evaluation experiments** from the project root using the paper experiments runner, for example:

```bash
python -m paper_experiments.runner \
  --agent tau2bench \
  --samples-file paper_experiments/datasets/tau2bench_dataset_easy.json \
  --user-proxy-persona expert
```

Additional options (agent type, datasets, number of trials, max turns, etc.) are documented in [paper_experiments/readme.md](paper_experiments/readme.md).

### Viewing Results

After running evaluations, results and error analysis are written to timestamped folders under `paper_experiments/`.
<!-- 
For each run, the [paper_experiments](paper_experiments/readme.md) runner creates an output directory such as `paper_experiments/experiment_outputs_<timestamp>/` containing, among others:

- `trial_<N>_results.json`: Per-trial, per-sample trajectories and metrics (AUC, PPT, turn counts, success flags).
- `aggregate_metrics_results.json`: Aggregate metrics (e.g., MaxAUC@k, MaxPPT@k) across all trials.
- `evaluation_results.pkl`: Serialized evaluation results.
- `error_analysis.pkl`: Serialized error analysis inputs and outputs used by the diagnosis UI.
- `evaluation.log`: Detailed logs for debugging and auditing. -->

See [paper_experiments/readme.md](paper_experiments/readme.md) for a full description of the output format.

### Error Diagnosis UI

To explore error analysis for a specific experiment run in a browser UI, you can launch the Streamlit viewer.

```bash
python -m streamlit run paper_experiments/view_results.py -- --output-dir paper_experiments/experiment_outputs_<timestamp>
```

Replace `<timestamp>` with the actual timestamp of your output directory. This loads the pickled results and starts a Streamlit app at `http://localhost:8501` that visualizes error categories and per-sample diagnostics. More details are in [paper_experiments/readme.md](paper_experiments/readme.md).

### Download Pre-computed Results from HuggingFace

We provide pre-computed experiment results on HuggingFace so you can explore the error diagnosis UI without running the full evaluation pipeline yourself.

**1. Install the HuggingFace `huggingface_hub` library** (if not already installed):

```bash
pip install huggingface_hub
```

**2. Download the dataset using the provided script:**

```bash
python paper_experiments/download_hf_dataset.py --output-dir <path-to-output-folder>
```

To download a specific file:

```bash
python paper_experiments/download_hf_dataset.py --filename <filename> --output-dir <path-to-output-folder>
```

See [paper_experiments/download_hf_dataset.py](paper_experiments/download_hf_dataset.py) for all available options (`--repo-id`, `--repo-type`, etc.).

**3. Run error diagnosis on the downloaded results:**

Once you have downloaded the results, you can launch the Error Diagnosis UI to explore the pre-computed error analysis. Point the Streamlit viewer at the downloaded output directory:

```bash
python -m streamlit run paper_experiments/view_results.py -- --output-dir <path-to-downloaded-results>
```

Example command:

```bash
python -m streamlit run paper_experiments/view_results.py -- --output-dir dataset/tau2bench/airline/gpt_4_1/expert
```

This loads the `error_analysis.pkl` file from the downloaded results and starts a Streamlit app at `http://localhost:8501` where you can interactively browse error categories and per-sample diagnostics.

If you want to re-run the error analysis programmatically on the downloaded data, you can do so in Python:

```python
from agent_inspect.tools import ErrorAnalysis
from agent_inspect.models.tools import ErrorAnalysisDataSample

# Load the downloaded data samples
error_analyzer = ErrorAnalysis(llm_client=client, max_workers=3)
error_analysis_result = error_analyzer.analyze_batch(error_analysis_data_samples)
```

## Bring Your Own Agent

You can plug in your own agentic system as long as it exposes a suitable interface and you can convert its interaction traces into the data structures expected by the metrics.

Typical steps:

1. **Define an adapter** that maps your agent's conversation / tool-calling traces into `AgentDialogueTrace`. Extend the `BaseAdapter` class in `agent_inspect.metrics.adapters` to implement this mapping.
2. **Define a session** that will orchestrate the connection of your agent to the evaluation framework. Extend the `BaseSession` class in `paper_experiments/session.py`.
3. **Create your dataset** of evaluation samples with the required subgoals.

The code in [paper_experiments](paper_experiments/readme.md) and [agent_runners](agent_runners/README.md) provides concrete examples you can follow when integrating a new agent.

### Creating your own evaluation dataset

After connecting your agent to our evaluation framework, you will need to define your `EvaluationSample`, which contains the subgoals and user proxy instruction.

We provide a helper in [paper_experiments/convert_to_data_sample.py](paper_experiments/convert_to_data_sample.py) to convert your JSON dataset into the `EvaluationSample` format expected by our framework. This helper assumes your dataset follows the schema below (array of samples):

```json
[
  {
    "id": "<string>",
    "input": [
      {
        "role": "user",
        "content": "<task description and instructions>",
        "terminating_condition": "<natural-language condition describing when the task is considered complete>"
      }
    ],
    "metadata": {
      "subgoals": [
        {
          "type": "<string>",          
          "details": "<natural-language subgoal describing expected agent behavior>",
          "turn": "<turn index or 'all'>"
        },
        ...
      ],
      "expected_tools": [
        "[{'tool_code': '<tool_name>(param1=value1, ...)', 'output': '$AnyValue'}]",
        "... additional tool specifications ..."
      ],
      "trace_type": "<string>"         
    },
    "target": "<optional expected response or list of responses>",
    "domain": "<string>"               
  }
]
```

Concretely, each element in the top-level array represents one evaluation sample. The `subgoals` array defines the `SubGoal` objects used during evaluation, `input[0].content` is used as the `user_instruction`, and `metadata.expected_tools` (if present) encodes expected tool calls that are mapped into `ExpectedToolCall` and `ToolInputParameter` objects.

Note: `expected_tools` is optional for now, in the future we plan to support tool call related metrics.

## Known Issues
<!-- You may simply state "No known issues. -->
No known issues.

## How to obtain support
[Create an issue](https://github.com/SAP/agent-quality-inspect/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).


## Contributing
Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## Citation

If you use this repository or the TED evaluation methodology in your research, please consider citing us:

```bibtex
@inproceedings{
  chong2026talk,
  title={Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis},
  author={Penny Chong and Harshavardhan Abichandani and Jiyuan Shen and Atin Ghosh and Min Pyae Moe and Yifan Mai and Daniel Dahlmeier},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=fHsVNklKOc}
}
```

## License
Copyright (c) 2026 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file. 

Disclaimer: This repository uses third‑party APIs that are subject to their own terms, fees, and compliance obligations.
