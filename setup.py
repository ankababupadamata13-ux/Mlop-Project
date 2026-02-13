from setuptools import setup, find_packages

# Read requirements.txt safely
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="mlops_project",
    version="0.1.0",
    author="Sudhanshu",
    description="Minor MLOps project using CI/CD pipeline",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
