import os
import subprocess
import sys
import re

# Function to check if a package is installed
def is_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Function to install a package using pip
def install_package(package_name):
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        print(f"{package_name} installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}.")

# Directory where your project scripts are located
project_directory = "/path/to/your/project"

# Regular expression to capture import statements
import_regex = re.compile(r'^\s*(?:import|from)\s+(\S+)', re.MULTILINE)

# Set to store packages
packages_needed = set()

# Walk through all files in the project directory
for root, dirs, files in os.walk(project_directory):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                found_imports = import_regex.findall(content)
                for imp in found_imports:
                    package_name = imp.split('.')[0]  # Get the top-level package
                    if not is_installed(package_name):
                        packages_needed.add(package_name)

# Install missing packages
for package in packages_needed:
    install_package(package)

if not packages_needed:
    print("All required packages are already installed.")
else:
    print(f"Installed packages: {', '.join(packages_needed)}")
