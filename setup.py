import setuptools
import os


# --- Helper function to read a file ---
def read_file(filename):
    """Read the content of a file."""
    # Determine the absolute path to the directory containing setup.py
    setup_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(setup_dir, filename)
    if not os.path.exists(file_path):
        print(f"Warning: File '{filename}' not found at '{file_path}'.")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# --- Helper function to read requirements ---
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    # Determine the absolute path to the directory containing setup.py
    setup_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(setup_dir, filename)
    if not os.path.exists(file_path):
        print(
            f"Warning: Requirements file '{filename}' not found at '{file_path}'. No dependencies will be loaded."
        )
        return []

    content = read_file(filename)  # Use the read_file helper
    lines = (line.strip() for line in content.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


# --- Read README.md for long description ---
long_description = read_file("README.md")
if not long_description:
    long_description = "CityVision: AI-powered traffic counting and analysis. See repository for details."

# --- Get the requirements from requirements.txt ---
requirements = parse_requirements("requirements.txt")
if not requirements:
    print("Warning: No requirements loaded. Check 'requirements.txt'.")

setuptools.setup(
    name="cityvision",
    author="CityVision Developers",
    author_email="sahand.somi@domain.com",
    description="AI-powered traffic counting and analysis library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        where="."
    ),  # Automatically find packages in the current directory
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        # Using the SPDX identifier for AGPLv3 as it's standard and precise.
        # The LICENSE file indicates "Version 3", not "or later".
        "License :: OSI Approved :: GNU Affero General Public License v3.0 (AGPL-3.0-only)",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: POC",
        "Natural Language :: English",
    ],
    python_requires=">=3.11",
)
