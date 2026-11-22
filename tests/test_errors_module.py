"""The public `cellestial.errors` module and the `CellestialError` base."""

import inspect

import pytest

import cellestial as cl
import cellestial.errors as errors
import cellestial.util.errors as implementation
from cellestial.errors import CellestialError, CellestialWarning, KeyNotFoundError

BUILTIN_EXCEPTIONS = (KeyError, ValueError, TypeError, IndexError, OSError)
ERROR_NAMES = [
    name for name in errors.__all__ if name not in {"CellestialError", "CellestialWarning"}
]


def test_module_is_reachable_from_package():
    assert cl.errors is errors
    assert cl.errors.KeyNotFoundError is KeyNotFoundError
    assert "KeyNotFoundError" not in cl.__all__


def test_all_lists_every_public_class():
    public = {
        name
        for name, member in vars(implementation).items()
        if inspect.isclass(member)
        and issubclass(member, (Exception, Warning))
        and member.__module__ == implementation.__name__
        and not name.startswith("_")
    }
    assert set(errors.__all__) == public


@pytest.mark.parametrize("name", errors.__all__)
def test_reexport_is_the_implementation_class(name):
    assert getattr(errors, name) is getattr(implementation, name)


@pytest.mark.parametrize("name", ERROR_NAMES)
def test_error_subclasses_base_and_builtin(name):
    error_class = getattr(errors, name)
    assert issubclass(error_class, CellestialError)
    assert issubclass(error_class, BUILTIN_EXCEPTIONS)
    # The builtin base comes first, so behaviour matches the builtin.
    assert error_class.__mro__.index(CellestialError) > min(
        error_class.__mro__.index(builtin)
        for builtin in BUILTIN_EXCEPTIONS
        if issubclass(error_class, builtin)
    )
    assert str(error_class("plain message")) == "plain message"


def test_warning_is_not_an_error():
    assert issubclass(CellestialWarning, UserWarning)
    assert not issubclass(CellestialWarning, CellestialError)


def test_spatial_library_error_removed():
    assert not hasattr(implementation, "SpatialLibraryError")


def test_real_call_caught_by_base_and_builtin(adata):
    with pytest.raises(CellestialError, match="leiden"):
        cl.umap(adata, "leidn")
    with pytest.raises(KeyError, match="leiden"):
        cl.umap(adata, "leidn")
    with pytest.raises(CellestialError) as caught:
        cl.umap(adata, "leidn")
    assert isinstance(caught.value, KeyNotFoundError)
    assert not str(caught.value).startswith("'")
