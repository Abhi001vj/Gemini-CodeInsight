# Gemini-CodeInsight

## Overview

CodeInsight is a tool that analyzes all code in a given directory, maps the relationships between files and their imports, and generates visualizations and a CSV of the graph nodes. It also leverages the Gemini API to answer complex questions about the codebase.

## Installation

To install the package, run:

```bash
pip install .
```

## Usage

```
codeinsight --directory /path/to/your/code --gemini-api-key YOUR_GEMINI_API_KEY
```
This will combine all code files into one, analyze the combined code, generate visualizations, and use the Gemini API to answer various questions about the codebase.

## Overview

This project analyzes all code in a given directory, maps the relationships between files and their imports, and generates visualizations and a CSV of the graph nodes. It also leverages the Gemini API to answer complex questions about the codebase.

## Structure

- `combine_code.sh`: Bash script to combine all code files into one.
- `analyze_code.py`: Python script to analyze the combined code file, generate a graph, visualize it, and export graph nodes to CSV.
- `cli.py`: CLI for running the analysis.
- `README.md`: Project documentation.

## Features

Codebase Statistics:

Lines of Code (LOC)
Number of Functions and Classes
Complexity Metrics (Cyclomatic Complexity)
Comment Density
Commit History Analysis:

Author Contributions
Frequency of Commits by Author
Time Series Analysis of Commits
Dependency Analysis:

Internal vs. External Dependencies
Visualization of Dependency Graph
File-Level Analysis:

Size of Each File
Most Changed Files
Documentation Quality:

Presence of Docstrings
Coverage of Documentation
Test Coverage:

Percentage of Code Covered by Tests
List of Test Files and Their Coverage
Code Smells and Anti-Patterns:

Identify Common Code Smells
Highlight Anti-Patterns
Security Analysis:

Potential Vulnerabilities
Use of Deprecated Libraries
Visualization Enhancements:

Interactive Visualizations (Using Plotly or Bokeh)
Word Clouds for Most Frequent Terms in Codebase
Heatmaps for Code Changes Over Time
Machine Learning Models:

Detailed Architecture Diagrams
Parameter Counts
Performance Metrics
Suggestions for Improvement:

Refactoring Suggestions
Potential Performance Optimizations
Suggested Libraries or Tools
Comparison with Similar Projects:

Comparison with Other Open Source Repositories
Benchmarking Against Industry Standards

Analyzing the Codebase
We'll create a Python script that analyzes the codebase, extracts information, and generates visualizations. This script will include all the necessary functions to process the codebase and produce visual outputs.

Generate Visualizations
We'll create various visualizations such as network graphs, bar charts, time series plots, and word clouds to illustrate different aspects of the codebase.


## Usage

1. Combine all code files into one:

    ```bash
    bash combine_code.sh /path/to/your/code
    ```

2. Analyze the combined code file and generate visualizations and CSV:

    ```python
    python analyze_code.py
    ```

3. Use the Gemini API to gain further insights:

    Update the `call_gemini_api` function with your Gemini API key and run the script.

## Future Enhancements

- Enhance the graph analysis to distinguish between internal and external modules.
- Improve visualization with interactive graphs.
- Integrate more detailed code analysis using additional tools and APIs.

## References

- [Gemini API Documentation](https://api.gemini.com/docs)
- [NetworkX Documentation](https://networkx.github.io/documentation/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Gemini Cookbook](https://github.com/google-gemini/cookbook?tab=readme-ov-file)
- [Gemini Cookbook Code execution](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Code_Execution.ipynb)
- [Gemini Cookbook Files API](https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_API.ipynb)
- [generative-ai-dart](https://github.com/google-gemini/generative-ai-dart)
- Gemini Cookbook Prompting](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb)
- [Gemini Prompt Gallery](https://ai.google.dev/gemini-api/prompts)

