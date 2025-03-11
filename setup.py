from setuptools import setup, find_packages

setup(
    name="langgfm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # install_requires=[
    #     # dependency
    #     # "numpy>=1.21.0",
    #     # "torch>=2.0.0"
    # ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for LangGFM project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/langgfm",  # optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)