from setuptools import find_packages, setup

setup(
    author="mizu-bai",
    description="Jax Permutational Invariant Polynomials",
    name="JaxPIP",
    packages=find_packages(
        include=[
            "jaxpip",
            "jaxpip.*",
        ],
    ),
    version="0.0.1",
)
