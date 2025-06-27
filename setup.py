from setuptools import setup, find_packages

setup(
    name="piczl",  # Replace with your package name
    version="0.1.0",
    author="William Roster",
    author_email="wroster@mpe.mpg.de",
    description="PICZL photometric redshifts.",
    packages=find_packages(where="src"),  # Specify 'src' as the source folder
    package_dir={"": "src"},  # Specify 'src' as the package root
    install_requires=[  # List your package dependencies here
        "tensorflow",
        "tensorflow_probability",
        "torch",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "PyYAML",
        "astropy",
    ],
    python_requires=">=3.6",  # Specify the minimum Python version
)
