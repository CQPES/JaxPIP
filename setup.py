from setuptools import find_packages, setup

setup(
    author="mizu-bai",
    description="Differentiable Permutation Invariant Polynomial (PIP) descriptor implemented in Jax.",
    name="JaxPIP",
    packages=find_packages(
        include=[
            "jaxpip",
            "jaxpip.*",
        ],
    ),
    version="0.0.1",
)
