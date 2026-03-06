import os
import re

def resolve_conflicts(directory):
    for root, dirs, files in os.walk(directory):
        if ".git" in root or ".venv" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "<<<<<<< HEAD" in content:
                    print(f"Resolving conflicts in {file_path}")
                    # Basic regex to pick the first block (HEAD side)
                    # Pattern: <<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [a-f0-9]+
                    pattern = r"<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [a-f0-9]+"
                    resolved_content = re.sub(pattern, r"\1", content, flags=re.DOTALL)
                    
                    # Also handle markers without newline just in case
                    pattern2 = r"<<<<<<< HEAD(.*?)\n=======(.*?)\n>>>>>>> [a-f0-9]+"
                    resolved_content = re.sub(pattern2, r"\1", resolved_content, flags=re.DOTALL)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(resolved_content)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    resolve_conflicts(".")
