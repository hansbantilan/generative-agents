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
        "langchain==0.0.142",
        "openai==0.27.4",
    ],
    extras_require={
        "mac": [],
        "linux": [
            "termcolor==2.2.0",
        ],
    },
)
