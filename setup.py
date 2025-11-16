#!/usr/bin/env python3
"""Setup script for GitView."""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gitview",
    version="0.1.0",
    author="GitView Contributors",
    author_email="",  # Add if publishing
    description="Git history analyzer with LLM-powered narrative generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carstenbund/gitview",
    project_urls={
        "Bug Reports": "https://github.com/carstenbund/gitview/issues",
        "Source": "https://github.com/carstenbund/gitview",
        "Documentation": "https://github.com/carstenbund/gitview/blob/main/README.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    keywords=[
        "git",
        "history",
        "analyzer",
        "llm",
        "narrative",
        "documentation",
        "ai",
        "claude",
        "openai",
        "ollama",
        "codebase",
        "evolution",
        "repository",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Software Development :: Documentation",
        "Topic :: Text Processing :: Markup :: Markdown",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gitview=gitview.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
