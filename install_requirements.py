import os
import ast
import subprocess

# Set the path to the directory you want to scan
directory = '/home/sc.uni-leipzig.de/ys09emuw/MACode'

# A set to store unique package names
packages = set()

# Function to scan a Python file for import statements
def scan_file(file_path):
    with open(file_path, 'r') as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        packages.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    packages.add(node.module)
        except Exception as e:
            print(f"Could not parse {file_path}: {e}")

# Scan each Python file in the directory
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.py'):
            scan_file(os.path.join(root, file))

# Create a list of packages to install
packages_to_install = list(packages)
if packages_to_install:
    print("Found the following packages:")
    for pkg in packages_to_install:
        print(f"- {pkg}")

    # Prompt the user to install the packages
    confirm = input("Do you want to install these packages? (yes/no): ")
    if confirm.lower() == 'yes':
        for pkg in packages_to_install:
            print(f"Installing {pkg}...")
            subprocess.run(['pip', 'install', pkg])
else:
    print("No packages found.")

