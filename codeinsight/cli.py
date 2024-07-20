import click
from .analyze_code import analyze_code

@click.command()
@click.option('--directory', '-d', required=True, type=click.Path(exists=True), help='Directory to analyze')
@click.option('--gemini-api-key', '-k', required=True, help='Gemini API key')
def main(directory, gemini_api_key):
    """CLI for CodeInsight"""
    analyze_code(directory, gemini_api_key)

if __name__ == '__main__':
    main()
