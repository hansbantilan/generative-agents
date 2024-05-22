from glob import glob

from setuptools import find_packages, setup

setup(
    name="generative-agents",
    version="0.1",
    packages=find_packages(),
    url="",
    license="",
    author="",
    author_email="",
    description="Generative Agents libraries",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    data_files=[
        ("workflows", [x for x in glob("workflows/**", recursive=True) if "." in x])
    ],
    install_requires=[
        "pre-commit==3.2.0",
        "langchain==0.2.0",
        "langchain-community==0.2.0",
        "openai==1.3.7",
        "anthropic==0.7.7",
        "pinecone-client==4.1.0",
        "termcolor==2.2.0",
        "faiss-cpu==1.7.4",
        "tiktoken==0.3.3",
        "streamlit==1.29.0",
        "streamlit-chat==0.1.1",
    ],
    extras_require={
        "mac": [],
        "linux": [],
    },
)
