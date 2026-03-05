import importlib.metadata
import sys

__version__ = importlib.metadata.version("cellestial")


def versions():
    """Print the in-use versions of the packages and python."""
    packages = [
        "cellestial",
        "scanpy",
        "anndata",
        "polars",
    ]

    text = ""
    for package in packages:
        text += f"{package:<12}: {importlib.metadata.version(package)}\n"
    text+=f"{"Python":12}: {sys.version.split()[0]}"
    print(text)


# test the function
if __name__ == "__main__":
    versions()
