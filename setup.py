from setuptools import setup, find_packages

setup(
    name="ai-memory-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.10",
    description="Local-first memory system for AI agents",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    license="MIT",
)
