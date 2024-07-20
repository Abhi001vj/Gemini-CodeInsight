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
from PIL import Image
import base64
from io import BytesIO
import logging
import traceback

def setup_logging(report_dir):
    log_dir = os.path.join(report_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'generate_reports.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This logs to console
        ]
    )

def log_error(step, file_path, exception):
    error_dir = os.path.join(report_dir, 'errors')
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)
    error_log_path = os.path.join(error_dir, f'error_{step}.log')
    with open(error_log_path, 'a', encoding='utf-8') as error_log:
        error_log.write(f"Error in step: {step}\n")
        error_log.write(f"File: {file_path}\n")
        error_log.write(f"Exception: {exception}\n")
        error_log.write(f"Traceback: {traceback.format_exc()}\n")
        error_log.write("\n")

def parse_imports(file_content):
    # Remove BOM if present
    file_content = file_content.lstrip('\ufeff')
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        logging.error(f"Syntax error while parsing content: {e}")
        raise e
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
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        log_error("build_graph", file_path, e)
        return None
    
    try:
        imports = parse_imports(content)
    except Exception as e:
        logging.error(f"Error parsing imports in file {file_path}: {e}")
        log_error("build_graph - parse_imports", file_path, e)
        return None
    
    G = nx.DiGraph()
    
    for imp in imports:
        imp_type = classify_import(imp, user_modules)
        file_stats = next((item for item in code_stats if item["file"].endswith(imp + ".py")), {})
        G.add_node(imp, type=imp_type, stats=file_stats)
        G.add_edge("main_file", imp)
    
    return G

def save_graph_to_csv(graph, output_path):
    try:
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
        logging.info(f"Graph saved to CSV at {output_path}")
    except Exception as e:
        logging.error(f"Error saving graph to CSV {output_path}: {e}")
        log_error("save_graph_to_csv", output_path, e)

def visualize_graph(graph, output_path):
    try:
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
        logging.info(f"Graph visualized and saved at {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing graph {output_path}: {e}")
        log_error("visualize_graph", output_path, e)

def get_commit_history(directory):
    try:
        os.chdir(directory)
        commit_history = os.popen('git log --pretty=format:"%h - %an, %ar : %s"').read().splitlines()
        os.chdir('..')
        logging.info(f"Commit history fetched for {directory}")
        return commit_history
    except Exception as e:
        logging.error(f"Error fetching commit history for {directory}: {e}")
        log_error("get_commit_history", directory, e)
        return []

def visualize_commit_history(commit_history, output_path):
    try:
        dates = [re.search(r'(\d+ \w+ \d+)', commit).group(1) for commit in commit_history]
        dates = [datetime.strptime(date, '%d %b %Y') for date in dates]
        date_counts = pd.Series(dates).value_counts().sort_index()
        
        fig = px.line(date_counts, title='Commit History Over Time', labels={'index': 'Date', 'value': 'Number of Commits'})
        fig.write_image(output_path)
        logging.info(f"Commit history visualized and saved at {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing commit history {output_path}: {e}")
        log_error("visualize_commit_history", output_path, e)

def analyze_codebase(directory):
    stats = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Remove BOM if present
                        if content.startswith('\ufeff'):
                            content = content[1:]
                        try:
                            raw_metrics = analyze(content)
                            complexity_metrics = cc_visit(content)
                            mi_score = mi_visit(content, False)
                            stats.append({
                                'file': file_path,
                                'loc': raw_metrics.loc,
                                'lloc': raw_metrics.lloc,
                                'sloc': raw_metrics.sloc,
                                'comments': raw_metrics.comments,
                                'multi': raw_metrics.multi,
                                'blank': raw_metrics.blank,
                                'complexity': sum(c.complexity for c in complexity_metrics),
                                'mi': mi_score,
                            })
                        except Exception as inner_e:
                            logging.error(f"Error analyzing content of file {file_path}: {inner_e}")
                            log_error("analyze_codebase - analyze content", file_path, inner_e)
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
                    log_error("analyze_codebase - read file", file_path, e)
    logging.info(f"Codebase analyzed for {directory}")
    return stats

def visualize_codebase_stats(stats, output_path):
    try:
        df = pd.DataFrame(stats)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['file'], y=df['loc'], name='Lines of Code'))
        fig.add_trace(go.Bar(x=df['file'], y=df['complexity'], name='Complexity'))
        fig.update_layout(barmode='group', title='Codebase Statistics', xaxis_title='File', yaxis_title='Count')
        fig.write_image(output_path)
        logging.info(f"Codebase statistics visualized and saved at {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing codebase statistics {output_path}: {e}")
        log_error("visualize_codebase_stats", output_path, e)

def call_gemini_api(prompt, gemini_api_key, uploaded_file, system_instruction=None):
    try:
        genai.configure(api_key=gemini_api_key)
        if system_instruction is None:
            system_instruction = """
            You are an experienced software developer assigned to review a new codebase. 
            Think as an expert programmer and analyze teh code accordingly with precision and care.
            Think about code best practices and dependencies and how this knowledge about code base can be shared to a fellow programmer. 
            """
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro", system_instruction=system_instruction)
        
        print(f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")  # Print file upload confirmation
        response = model.generate_content(
            [prompt, uploaded_file]
        )
        logging.info(f"Gemini API called with prompt: {prompt}")
        print(f"Gemini API response received for prompt: {prompt}")  # Print prompt processing confirmation
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini API with prompt {prompt}: {e}")
        log_error("call_gemini_api", "Gemini API call", e)
        print(f"Error calling Gemini API with prompt {prompt}: {e}")  # Print error message
        return ""

def decode_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        log_error("decode_image", "Image decoding", e)
        print(f"Error decoding image: {e}")  # Print error message
        return None

def generate_reports(directory, gemini_api_key):
    global report_dir
    report_dir = f'{directory}_report'
    setup_logging(report_dir)

    logging.info("Combining code files...")
    try:
        os.system(f'bash codeinsight/combine_code.sh {directory}')
        logging.info(f"Code files combined into combined_code_dump.txt for {directory}")
    except Exception as e:
        logging.error(f"Error combining code files: {e}")
        log_error("generate_reports - combine_code.sh", directory, e)
        print(f"Error combining code files: {e}")  # Print error message

    # Directory structure for Gemini results and local Python results
    gemini_results_dir = os.path.join(report_dir, 'gemini_results')
    local_results_dir = os.path.join(report_dir, 'local_results')
    if not os.path.exists(gemini_results_dir):
        os.makedirs(gemini_results_dir)
        os.makedirs(os.path.join(gemini_results_dir, 'visualizations'))
        os.makedirs(os.path.join(gemini_results_dir, 'csv'))
        os.makedirs(os.path.join(gemini_results_dir, 'text'))
    if not os.path.exists(local_results_dir):
        os.makedirs(local_results_dir)
        os.makedirs(os.path.join(local_results_dir, 'visualizations'))
        os.makedirs(os.path.join(local_results_dir, 'csv'))
        os.makedirs(os.path.join(local_results_dir, 'text'))
    
    # Grouped Gemini Queries
    graph_queries = [
        "Draw me an overall diagram of the major modules of the system.",
        "Draw me neural net architectures for any sub models used.",
        "Distinguish between internal vs. external dependencies."
    ]
    
    csv_queries = [
        "Give me a list of files and summaries of what they do. Return the response as CSV with columns: File Name, Summary.",
        "List the test files and their coverage. Return the response as CSV with columns: Test File Name, Coverage Percentage.",
        "Identify common code smells. Return the response as CSV with columns: File Name, Code Smell, Description."
    ]
    
    text_prompt = """
    Please generate a detailed markdown report for the provided codebase covering the following aspects:
    - List external components used.
    - Describe the future roadmap of possible enhancements.
    - Recommend readings to better understand the ideas behind the code.
    - List other open source repositories similar to this one.
    - Analyze author contributions with a table of commit counts by author.
    - Show the frequency of commits by each author in a table.
    - Provide a time series analysis of commits.
    - Identify potential vulnerabilities with code examples. Provide detailed explanations and code highlighting.
    - Highlight the use of deprecated libraries with code examples. Provide detailed explanations and code highlighting.
    - Suggest refactoring with before and after code highlighting. Provide detailed explanations and code highlighting.
    - Suggest potential performance optimizations with before and after code highlighting. Provide detailed explanations and code highlighting.
    - Recommend libraries or tools that could be beneficial.
    - Compare with other open source repositories.
    - Benchmark against industry standards.
    """
    uploaded_file = genai.upload_file(path="combined_code_dump.txt", display_name="Repository code")
    # Fetch Graph Data
    for i, query in enumerate(graph_queries):
        response = call_gemini_api(query, gemini_api_key,uploaded_file)
        if response.startswith("data:image"):
            image_data = response.split(",")[1]
            image = decode_image(image_data)
            if image:
                try:
                    image.save(os.path.join(gemini_results_dir, 'visualizations', f'graph_response_{i}.png'))
                    logging.info(f"Graph response saved as image at {os.path.join(gemini_results_dir, 'visualizations', f'graph_response_{i}.png')}")
                    print(f"Graph response saved as image at {os.path.join(gemini_results_dir, 'visualizations', f'graph_response_{i}.png')}")  # Print confirmation
                except Exception as e:
                    logging.error(f"Error saving graph response image {os.path.join(gemini_results_dir, 'visualizations', f'graph_response_{i}.png')}: {e}")
                    log_error("generate_reports - save graph image", f'graph_response_{i}.png', e)
                    print(f"Error saving graph response image {os.path.join(gemini_results_dir, 'visualizations', f'graph_response_{i}.png')}: {e}")  # Print error message
        else:
            try:
                with open(os.path.join(gemini_results_dir, 'text', f'graph_response_{i}.txt'), 'w') as f:
                    f.write(response)
                logging.info(f"Graph response saved as text at {os.path.join(gemini_results_dir, 'text', f'graph_response_{i}.txt')}")
                print(f"Graph response saved as text at {os.path.join(gemini_results_dir, 'text', f'graph_response_{i}.txt')}")  # Print confirmation
            except Exception as e:
                logging.error(f"Error saving graph response text {os.path.join(gemini_results_dir, 'text', f'graph_response_{i}.txt')}: {e}")
                log_error("generate_reports - save graph text", f'graph_response_{i}.txt', e)
                print(f"Error saving graph response text {os.path.join(gemini_results_dir, 'text', f'graph_response_{i}.txt')}: {e}")  # Print error message

    # Fetch CSV Data
    for query, filename in zip(csv_queries, ['files_summaries.csv', 'test_files_coverage.csv', 'common_code_smells.csv']):
        response = call_gemini_api(query, gemini_api_key)
        try:
            with open(os.path.join(gemini_results_dir, 'csv', filename), 'w') as f:
                f.write(response)
            logging.info(f"CSV response saved at {os.path.join(gemini_results_dir, 'csv', filename)}")
            print(f"CSV response saved at {os.path.join(gemini_results_dir, 'csv', filename)}")  # Print confirmation
        except Exception as e:
            logging.error(f"Error saving CSV response {os.path.join(gemini_results_dir, 'csv', filename)}: {e}")
            log_error("generate_reports - save csv", filename, e)
            print(f"Error saving CSV response {os.path.join(gemini_results_dir, 'csv', filename)}: {e}")  # Print error message

    # Fetch Text Data
    text_response = call_gemini_api(text_prompt, gemini_api_key)
    
    # Save Text Response
    try:
        with open(os.path.join(gemini_results_dir, 'text', 'gemini_commentary.md'), 'w') as f:
            f.write(text_response)
        logging.info("Text response saved at gemini_results_dir/text/gemini_commentary.md")
        print("Text response saved at gemini_results_dir/text/gemini_commentary.md")  # Print confirmation
    except Exception as e:
        logging.error(f"Error saving text response {os.path.join(gemini_results_dir, 'text', 'gemini_commentary.md')}: {e}")
        log_error("generate_reports - save text response", 'gemini_commentary.md', e)
        print(f"Error saving text response {os.path.join(gemini_results_dir, 'text', 'gemini_commentary.md')}: {e}")  # Print error message
    
    # Generate Local Reports
    user_modules = get_user_modules(directory)
    code_stats = analyze_codebase(directory)
    graph = build_graph("combined_code_dump.txt", user_modules, code_stats)
    if graph:
        save_graph_to_csv(graph, os.path.join(local_results_dir, 'csv', 'graph_nodes.csv'))
        visualize_graph(graph, os.path.join(local_results_dir, 'visualizations', 'import_graph.png'))
    
    commit_history = get_commit_history(directory)
    visualize_commit_history(commit_history, os.path.join(local_results_dir, 'visualizations', 'commit_history.png'))
    
    visualize_codebase_stats(code_stats, os.path.join(local_results_dir, 'visualizations', 'codebase_stats.png'))
    
    generate_summary_report(commit_history, code_stats, local_results_dir)

def generate_summary_report(commit_history, code_stats, report_dir):
    try:
        with open(os.path.join(report_dir, 'summary_report.md'), 'w') as report_file:
            report_file.write("# Code Repository Analysis Summary Report\n\n")
            
            report_file.write("## Import Graph\n")
            report_file.write("![Import Graph](visualizations/import_graph.png)\n\n")
            
            report_file.write("## Commit History\n")
            report_file.write("![Commit History](visualizations/commit_history.png)\n\n")
            
            report_file.write("## Codebase Statistics\n")
            report_file.write("![Codebase Statistics](visualizations/codebase_stats.png)\n\n")
            
            report_file.write("## Commit History Details\n")
            report_file.write("A time series graph showing the commit history over time has been generated.\n")
        logging.info("Summary report generated at report/summary_report.md")
        print("Summary report generated at report/summary_report.md")  # Print confirmation
    except Exception as e:
        logging.error(f"Error generating summary report {os.path.join(report_dir, 'summary_report.md')}: {e}")
        log_error("generate_summary_report", 'summary_report.md', e)
        print(f"Error generating summary report {os.path.join(report_dir, 'summary_report.md')}: {e}")  # Print error message

if __name__ == '__main__':
    import sys
    directory = sys.argv[1]
    report_dir = f'{directory}_report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        os.makedirs(os.path.join(report_dir, 'logs'))
        os.makedirs(os.path.join(report_dir, 'errors'))
    
    gemini_api_key = sys.argv[2]
    generate_reports(directory, gemini_api_key)
