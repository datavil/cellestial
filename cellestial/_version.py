import importlib.metadata

__version__ = importlib.metadata.version("cellestial")


def versions():
    packages = [
        "cellestial",
        "scanpy",
        "anndata",
        "polars",
    ]

    text = ""
    for package in packages:
        text += f"{package}: {importlib.metadata.version(package)}\n"

    print(text)


# test the function
if __name__ == "__main__":
    versions()
