import ast
import networkx as nx
import csv
import matplotlib.pyplot as plt
import requests
import os
import google.generativeai as genai
import pandas as pd
import json
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
from collections import Counter

def parse_imports(file_content):
    tree = ast.parse(file_content)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    return imports

def classify_import(import_name, user_modules):
    if import_name in user_modules:
        return 'user_module'
    elif import_name in sys.builtin_module_names:
        return 'internal_module'
    else:
        return 'external_library'

def get_user_modules(directory):
    user_modules = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                module_name = os.path.splitext(file)[0]
                user_modules.add(module_name)
    return user_modules

def build_graph(file_path, user_modules, code_stats):
    with open(file_path, 'r') as file:
        content = file.read()
    
    imports = parse_imports(content)
    G = nx.DiGraph()
    
    for imp in imports:
        imp_type = classify_import(imp, user_modules)
        file_stats = next((item for item in code_stats if item["file"].endswith(imp + ".py")), {})
        G.add_node(imp, type=imp_type, stats=file_stats)
        G.add_edge("main_file", imp)
    
    return G

def save_graph_to_csv(graph, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Source', 'Target', 'Type', 'LOC', 'Complexity', 'MI']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for edge in graph.edges():
            node = graph.nodes[edge[1]]
            writer.writerow({
                'Source': edge[0],
                'Target': edge[1],
                'Type': node.get('type'),
                'LOC': node.get('stats', {}).get('loc', 'N/A'),
                'Complexity': node.get('stats', {}).get('complexity', 'N/A'),
                'MI': node.get('stats', {}).get('mi', 'N/A')
            })

def visualize_graph(graph, output_path):
    color_map = {
        'user_module': 'skyblue',
        'internal_module': 'lightgreen',
        'external_library': 'lightcoral'
    }

    node_colors = [color_map[graph.nodes[node]['type']] for node in graph.nodes]

    plt.figure(figsize=(10, 10))
    nx.draw(graph, with_labels=True, node_color=node_colors, node_size=1500, edge_color='gray')
    plt.savefig(output_path)
    plt.close()

def get_commit_history(directory):
    os.chdir(directory)
    commit_history = os.popen('git log --pretty=format:"%h - %an, %ar : %s"').read().splitlines()
    os.chdir('..')
    return commit_history

def visualize_commit_history(commit_history, output_path):
    dates = [re.search(r'(\d+ \w+ \d+)', commit).group(1) for commit in commit_history]
    dates = [datetime.strptime(date, '%d %b %Y') for date in dates]
    date_counts = pd.Series(dates).value_counts().sort_index()
    
    fig = px.bar(date_counts, title='Commit History', labels={'index': 'Date', 'value': 'Number of Commits'})
    fig.write_image(output_path)

def analyze_codebase(directory):
    stats = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    raw_metrics = analyze(content)
                    complexity_metrics = cc_visit(content)
                    mi_metrics = mi_visit(content)
                    stats.append({
                        'file': file_path,
                        'loc': raw_metrics.loc,
                        'lloc': raw_metrics.lloc,
                        'sloc': raw_metrics.sloc,
                        'comments': raw_metrics.comments,
                        'multi': raw_metrics.multi,
                        'blank': raw_metrics.blank,
                        'complexity': sum(c.complexity for c in complexity_metrics),
                        'mi': mi_metrics.mi,
                    })
    return stats

def visualize_codebase_stats(stats, output_path):
    df = pd.DataFrame(stats)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['file'], y=df['loc'], name='Lines of Code'))
    fig.add_trace(go.Bar(x=df['file'], y=df['complexity'], name='Complexity'))
    fig.update_layout(barmode='group', title='Codebase Statistics', xaxis_title='File', yaxis_title='Count')
    fig.write_image(output_path)

def call_gemini_api(question, gemini_api_key, combined_code_path):
    genai.configure(api_key=gemini_api_key)
    md_file = genai.upload_file(path=combined_code_path, display_name="Combined Code Dump", mime_type="text/plain")
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content([question, md_file])
    return response.text

def analyze_code(directory, gemini_api_key):
    os.system(f'bash codeinsight/combine_code.sh {directory}')
    
    user_modules = get_user_modules(directory)
    
    code_stats = analyze_codebase(directory)
    
    graph = build_graph("combined_code_dump.txt", user_modules, code_stats)
    save_graph_to_csv(graph, 'report/graph_nodes.csv')
    visualize_graph(graph, 'report/import_graph.png')
    
    commit_history = get_commit_history(directory)
    visualize_commit_history(commit_history, 'report/commit_history.png')
    
    visualize_codebase_stats(code_stats, 'report/codebase_stats.png')
    
    questions = """
    1. Draw me an overall diagram of the major modules of the system.
    2. Draw me neural net architectures for any sub models used.
    3. Give me a list of files and summaries of what they do.
    4. List external components used.
    5. Describe future roadmap of possible enhancements.
    6. What should I read to better understand the ideas behind the code?
    7. What are other open source repositories similar to this one?
    8. Lines of Code (LOC).
    9. Number of Functions and Classes.
    10. Cyclomatic Complexity.
    11. Comment Density.
    12. Author Contributions.
    13. Frequency of Commits by Author.
    14. Time Series Analysis of Commits.
    15. Internal vs. External Dependencies.
    16. Size of Each File.
    17. Most Changed Files.
    18. Presence of Docstrings.
    19. Coverage of Documentation.
    20. Percentage of Code Covered by Tests.
    21. List of Test Files and Their Coverage.
    22. Identify Common Code Smells.
    23. Highlight Anti-Patterns.
    24. Potential Vulnerabilities.
    25. Use of Deprecated Libraries.
    26. Detailed Architecture Diagrams for ML models.
    27. Parameter Counts for ML models.
    28. Performance Metrics for ML models.
    29. Refactoring Suggestions.
    30. Potential Performance Optimizations.
    31. Suggested Libraries or Tools.
    32. Comparison with Other Open Source Repositories.
    33. Benchmarking Against Industry Standards.
    """
    
    response = call_gemini_api(questions, gemini_api_key, "combined_code_dump.txt")
    
    generate_markdown_report(response, commit_history, code_stats)

def generate_markdown_report(gemini_response, commit_history, code_stats):
    with open('report/report.md', 'w') as report_file:
        report_file.write("# Code Repository Analysis Report\n\n")
        
        report_file.write("## Import Graph\n")
        report_file.write("![Import Graph](import_graph.png)\n\n")
        
        report_file.write("## Commit History\n")
        report_file.write("![Commit History](commit_history.png)\n\n")
        
        report_file.write("## Codebase Statistics\n")
        report_file.write("![Codebase Statistics](codebase_stats.png)\n\n")
        
        report_file.write("## Gemini Commentary\n")
        report_file.write(gemini_response + "\n\n")
        
        report_file.write("## Commit History Details\n")
        for commit in commit_history:
            report_file.write(f"{commit}\n")

if __name__ == '__main__':
    import sys
    if not os.path.exists('report'):
        os.makedirs('report')
    directory = sys.argv[1]
    gemini_api_key = sys.argv[2]
    analyze_code(directory, gemini_api_key)
