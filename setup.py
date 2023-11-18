from setuptools import find_packages, setup

setup(
    author="mizu-bai",
    description="Jax Implementation of Permutation Invariant Polynomial descriptor.",
    name="JaxPIP",
    packages=find_packages(
        include=[
            "jaxpip",
            "jaxpip.*",
        ],
    ),
    version="0.0.1",
)
