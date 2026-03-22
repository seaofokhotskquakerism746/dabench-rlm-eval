"""RLM solver for DABench data analysis tasks.

This module defines how the RLM solves each data analysis question over a CSV file.
GEPA evolves this entire file — the prompt, signature, parameters, and helper tools.

Contract:
  - Must define: run_task(question, constraints, format_spec, csv_path, verbose=False) -> str
  - `dspy` and `DataFrame` are available as globals
  - `csv_path` is the absolute path to a CSV file (loaded here, passed as DataFrame to RLM)
  - Return value must contain @field[value] formatted answers
"""

import pandas as pd

# -- RLM Prompt Template --
# Placeholders {inputs}, {output_fields}, {final_output_names}, {max_llm_calls}
# are filled by dspy.RLM at runtime — they MUST remain in the template.

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment with pandas, numpy, scipy, sklearn, and statistics available.
Write Python code and it will be executed. You will see the output, then write more code. This is iterative.

Available:
- Variables: {inputs} (your input data — `data` is a pandas DataFrame already loaded)
- `llm_query(prompt)` - query a sub-LLM for semantic analysis
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, statistics, etc.
- Data libraries: pandas, numpy, scipy, sklearn

IMPORTANT: This is ITERATIVE. Each code block executes, you see the output, then decide next steps.

IMPORTANT: When you need a library (sklearn, scipy, etc.), import it at the TOP LEVEL of your code block — never inside try/except. The sandbox auto-installs packages when it sees top-level imports.

Workflow:
1. EXPLORE - Inspect the DataFrame. Print data.head(), data.columns, data.dtypes, data.shape. Import any libraries you'll need (sklearn, scipy, etc.) at the top level.
2. UNDERSTAND - Read the question and constraints carefully. Identify which columns matter.
3. COMPUTE - Write code to answer the question step by step. Print intermediate results.
4. FORMAT - Format your answer exactly as specified in the format_spec using @field[value] notation.
5. VERIFY - Check your answer makes sense before submitting.
6. SUBMIT - Call SUBMIT() with your formatted answer string.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""


# -- Task Signature --

class DataAnalysisTask(dspy.Signature):
    """You are a data analyst. Given a dataset and a question, write Python code
    to analyze the data and produce the answer.

    The `data` variable is a pandas DataFrame already loaded in memory.
    Read the constraints carefully for methodology requirements.
    Format your answer exactly as specified in format_spec using @field[value] notation.
    """

    data: DataFrame = dspy.InputField(desc="The dataset as a pandas DataFrame")
    question: str = dspy.InputField(desc="The data analysis question to answer")
    constraints: str = dspy.InputField(desc="Methodology constraints and requirements")
    format_spec: str = dspy.InputField(desc="Required answer format using @field[value] notation")
    answer: str = dspy.OutputField(desc="The answer formatted per format_spec")


# -- RLM Configuration --

MAX_ITERATIONS = 15
MAX_LLM_CALLS = 30
MAX_OUTPUT_CHARS = 10_000


# -- Main Entry Point --

def run_task(question: str, constraints: str, format_spec: str, csv_path: str, verbose: bool = False) -> str:
    """Run the RLM to answer a single DABench task. Returns the response string."""
    import dspy.predict.rlm as rlm_module

    original = rlm_module.ACTION_INSTRUCTIONS_TEMPLATE
    rlm_module.ACTION_INSTRUCTIONS_TEMPLATE = ACTION_INSTRUCTIONS_TEMPLATE

    try:
        data = DataFrame(pd.read_csv(csv_path))
        rlm = dspy.RLM(
            DataAnalysisTask,
            max_iterations=MAX_ITERATIONS,
            max_llm_calls=MAX_LLM_CALLS,
            max_output_chars=MAX_OUTPUT_CHARS,
            verbose=verbose,
        )
        result = rlm(
            data=data,
            question=question,
            constraints=constraints,
            format_spec=format_spec,
        )
        answer = str(result.answer).strip()
        iterations = len(result.trajectory) if hasattr(result, 'trajectory') else None
        return {"answer": answer, "iterations": iterations}
    finally:
        rlm_module.ACTION_INSTRUCTIONS_TEMPLATE = original
