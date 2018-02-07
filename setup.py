from setuptools import find_packages, setup

requires = [
    "bottle", "matplotlib", "networkx", "numpy", "pandas", "scikit-learn", "scipy", "future"
]

setup(
    install_requires=requires,
    name="renom_tda",
    version="2.0.5",
    packages=find_packages(),
)
