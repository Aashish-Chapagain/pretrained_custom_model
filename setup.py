from setuptools import find_packages, setup

setup(
    name="minillm",
    version="0.1.0",
    description="A custom lightweight 6M parameter causal Transformer LLM",
    author="Custom LLM Project",
    packages=find_packages(),
    py_modules=["model", "generate", "chat", "datasetclass"],
    install_requires=[
        "torch",
        "sentencepiece",
        "numpy",
    ],
    python_requires=">=3.9",
)
