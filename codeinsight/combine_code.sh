#!/bin/bash

# Method from @deedydas on x.com https://x.com/deedydas/status/1802529135530856925
# Directory of the repository (default to current directory if not specified)
REPO_DIR="${1:-.}"

# Output file
OUTPUT_FILE="combined_code_dump.txt"

# List of file extensions to include
FILE_EXTENSIONS=("py" "js" "html" "css" "java" "cpp" "h" "cs")  # Add more extensions as needed

# Empty the output file if it exists
> "$OUTPUT_FILE"

# Function to combine files
combine_files() {
    local dir="$1"
    for ext in "${FILE_EXTENSIONS[@]}"; do
        find "$dir" -type f -name "*.$ext" -print0 | while IFS= read -r -d '' file; do
            echo "// File: $file" >> "$OUTPUT_FILE"
            cat "$file" >> "$OUTPUT_FILE"
            echo -e "\n\n" >> "$OUTPUT_FILE"
        done
    done
}

# Combine the files
combine_files "$REPO_DIR"

echo "All code files have been combined into $OUTPUT_FILE"
