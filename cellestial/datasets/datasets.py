import re
import shutil
from os import PathLike
from pathlib import Path

from anndata import AnnData, read_h5ad

_GLOBAL_CACHE = Path.home() / ".cache" / "cellestial" / "datasets"

_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_DOWNLOAD_TIMEOUT_SECONDS = 300

def _resolve_cache_file(
    cache_directory: str | Path | PathLike,
    filename: str,
    *,
    use_cache: bool,
    bring: bool,
) -> Path:
    """
    Prepare the cache directory and return the target cache file path.

    Creates `cache_directory` if missing. When `bring` is True and the user
    points at a non-default directory that doesn't yet contain `filename`,
    copies the file from `_GLOBAL_CACHE` if it exists there. When `use_cache`
    is False, removes the target cache file so it will be regenerated.
    """
    if not isinstance(cache_directory, Path):
        cache_directory = Path(cache_directory)

    if not cache_directory.exists():
        cache_directory.mkdir(parents=True)

    cache_file = cache_directory / filename

    if use_cache and bring and cache_directory != _GLOBAL_CACHE and not cache_file.exists():
        global_cache_file = _GLOBAL_CACHE / filename
        if global_cache_file.exists():
            print(f"Bringing cached file from {global_cache_file} to {cache_file}")
            shutil.copy(global_cache_file, cache_file)

    if not use_cache and cache_file.exists():
        print(f"Removing existing cache: {cache_file}")
        cache_file.unlink()

    return cache_file


def pbmc3k(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load the preprocessed pbmc3k dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and re-download.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed pbmc3k dataset.

    Raises
    ------
    OSError
        If the downloaded file size does not match the expected size.
    """
    return from_url(
        "https://huggingface.co/datasets/datavil/pbmc3k/resolve/main/pbmc3k_pped.h5ad",
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def pancreas(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load the preprocessed endocrinogenesis day-15 pancreas dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and re-download.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed pancreas dataset with velocity computed.

    Raises
    ------
    OSError
        If the downloaded file size does not match the expected size.
    """
    return from_url(
        "https://huggingface.co/datasets/datavil/pancreas/resolve/main/endocrinogenesis_day15_pped.h5ad",
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def from_url(
    url: str,
    *,
    name: str | None = None,
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load an `.h5ad` dataset from a direct HTTP(S) URL.

    Parameters
    ----------
    url : str
        Direct URL to a downloadable `.h5ad` file.
    name : str, optional
        Name for the cached file (without extension). If omitted, the cache
        filename is derived from the URL's last path segment.
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The downloaded dataset.

    Raises
    ------
    ValueError
        If `name` is not given and no filename can be derived from `url`.
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Generic loader for any host that serves an AnnData `.h5ad` file at a
    stable URL (CELLxGENE, Zenodo, Figshare, S3 buckets, etc.).
    """
    from urllib.parse import urlparse
    from urllib.request import urlopen

    from tqdm import tqdm

    if name is not None:
        filename = f"{name}.h5ad"
    else:
        filename = Path(urlparse(url).path).name
        if not filename:
            message = "Could not derive a filename from the URL; pass `name=...` explicitly."
            raise ValueError(message)

    cache_file = _resolve_cache_file(cache_directory, filename, use_cache=use_cache, bring=bring)

    if cache_file.exists():
        return read_h5ad(cache_file)

    print(f"No cache at {cache_file} !")
    print(f"Downloading from {url} ...")

    block_size = 1024 * 1024
    # Stream to a .part file and atomically rename on success, so an interrupted
    # download cannot poison the cache.
    partial_file = cache_file.with_name(cache_file.name + ".part")

    with urlopen(url, timeout = _DOWNLOAD_TIMEOUT_SECONDS) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        progress_bar = tqdm(
            total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {filename}"
        )

        try:
            with partial_file.open("wb") as file:
                while chunk := response.read(block_size):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

            if total_size != 0 and progress_bar.n != total_size:
                message = (
                    f"Download incomplete: expected {total_size} bytes, got {progress_bar.n}."
                )
                raise OSError(message)  # noqa: TRY301

            partial_file.replace(cache_file)
        except BaseException:
            if partial_file.exists():
                partial_file.unlink()
            raise
        finally:
            progress_bar.close()

    print(f"Saved to {cache_file}")

    return read_h5ad(cache_file)


def from_cellxgene(
    source: str,
    *,
    name: str | None = None,
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load a dataset from CELLxGENE by URL or UUID.

    Parameters
    ----------
    source : str
        A CELLxGENE dataset UUID (e.g.
        `7cdea341-ca7a-40fd-8192-b8ecb2d7b91e`) or a full `.h5ad` URL (e.g.
        `https://datasets.cellxgene.cziscience.com/<uuid>.h5ad`).
    name : str, optional
        Name for the cached file (without extension). Defaults to the dataset
        UUID parsed from `source`.
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The downloaded dataset.

    Raises
    ------
    ValueError
        If `source` is neither a CELLxGENE dataset UUID nor a URL containing one.
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Thin wrapper around `from_url` that accepts a CELLxGENE dataset UUID
    directly and constructs the canonical download URL.
    """
    match = _UUID_PATTERN.search(source)
    if match is None:
        message = (
            f"source must be a CELLxGENE dataset UUID or a URL containing one, got {source!r}"
        )
        raise ValueError(message)

    uuid = match.group(0)
    url = (
        source
        if source.startswith(("http://", "https://"))
        else f"https://datasets.cellxgene.cziscience.com/{uuid}.h5ad"
    )

    return from_url(
        url,
        name=name or uuid,
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def breast_cancer_atlas(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load the breast cancer cell atlas from CELLxGENE.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and regenerate from scratch.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The breast cancer atlas dataset.

    Raises
    ------
    OSError
        If the downloaded file size does not match the expected size.

    Notes
    -----
    Thin wrapper around `from_cellxgene` with the atlas UUID baked in.
    """
    return from_cellxgene(
        "7cdea341-ca7a-40fd-8192-b8ecb2d7b91e",
        name="breast_cancer_atlas",
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )


def human_lymph_node(
    cache_directory: str | Path | PathLike = _GLOBAL_CACHE,
    *,
    use_cache: bool = True,
    bring: bool = True,
) -> AnnData:
    """
    Download and load the preprocessed V1 Human Lymph Node Visium spatial dataset.

    Parameters
    ----------
    cache_directory : str | Path | PathLike
        Directory where the `.h5ad` file is cached.
    use_cache : bool, default=True
        If True, load from the cached file when present. If False, remove any
        existing cached file and re-download.
    bring : bool, default=True
        If True and `cache_directory` is not the default global cache, copy the
        file from the global cache when it already exists there. Set to False
        to disable this behavior.

    Returns
    -------
    AnnData
        The preprocessed V1 Human Lymph Node Visium dataset with clusters and UMAP.

    Raises
    ------
    OSError
        If the downloaded file size does not match the expected size.
    """
    return from_url(
        "https://huggingface.co/datasets/datavil/human_lymph_node/resolve/main/V1_Human_Lymph_Node_pped.h5ad",
        cache_directory=cache_directory,
        use_cache=use_cache,
        bring=bring,
    )
