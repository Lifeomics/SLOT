from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        reqs = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return reqs
setup(
    name="SLOT",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    author="Devinjzhu",
    author_email="zhuj21@mail.tsinghua.edu.cn",
    description="A package for Subcellular Location Optimal Transport (SLOT) algorithm.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)