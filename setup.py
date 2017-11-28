from setuptools import find_packages, setup

requires = [
    "bottle", "matplotlib", "networkx", "numpy", "pandas", "scikit-learn", "scipy", "future"
]

setup(
    install_requires=requires,
    name="renom_tda",
    version="1.0.0",
    packages=find_packages(),
)