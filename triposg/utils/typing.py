"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic types
from typing import (
    Any,
    TypedDict,
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md

# Config type

# PyTorch Tensor type

# Runtime type checking decorator


# Custom types
class FuncArgs(TypedDict):
    """Type for instantiating a function with keyword arguments"""

    name: str
    kwargs: dict[str, Any]

    @staticmethod
    def validate(variable):
        necessary_keys = ["name", "kwargs"]
        for key in necessary_keys:
            assert key in variable, f"Key {key} is missing in {variable}"
        if not isinstance(variable["name"], str):
            raise TypeError(
                f"Key 'name' should be a string, not {type(variable['name'])}"
            )
        if not isinstance(variable["kwargs"], dict):
            raise TypeError(
                f"Key 'kwargs' should be a dictionary, not {type(variable['kwargs'])}"
            )
        return variable
