from cellestial import dimensional, dimensionals, umap

def _remove_docstring_param(func, to_remove:str)-> str:
    # get the lines of the docstring into a list
    lines = func.__doc__.splitlines()
    # find the index of the parameter to remove
    index_to_remove = [i for i,line in enumerate(lines) if line.startswith(f"    {to_remove}")][0]
    # find the index of all possible parameters
    index_all = [i for i,line in enumerate(lines) if line.startswith("    ") and not line.startswith("        ")]
    # find the index of the next parameter
    next_at = index_all.index(index_to_remove) + 1
    index_next = index_all[next_at]
    # remove the lines between the parameter to remove and the next parameter from the list
    new_lines = [line for i,line in enumerate(lines) if i not in range(index_to_remove,index_next)]
    # convert the list back to a string and return it
    new_doc = '\n'.join(new_lines)
    return new_doc


subdimensional_docstring = _remove_docstring_param(dimensional, "dimensions")
dimensional_grid_docstring = _remove_docstring_param(dimensionals, "dimensions")
print(subdimensional_docstring)

def test():
    test.__doc__ = subdimensional_docstring
    print(test.__doc__)
    return

umap.__doc__ = dimensional_grid_docstring

a = umap()