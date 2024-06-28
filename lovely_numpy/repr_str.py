# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_repr_str.ipynb.

# %% auto 0
__all__ = ['lovely']

# %% ../nbs/00_repr_str.ipynb 4
from typing import Union, Optional as O
from collections import defaultdict
from fastcore.foundation import store_attr
import warnings
import numpy as np

from .utils import pretty_str, sparse_join, np_to_str_common, in_debugger, bytes_to_human
from .utils.config import get_config, set_config, config

# %% ../nbs/00_repr_str.ipynb 6
dtnames =   {   "float16": "f16",
                "float32": "f32",
                "float64": "", # Default dtype in numpy
                "uint8": "u8",
                "uint16": "u16",
                "uint32": "u32",
                "uint64": "u64",
                "int8": "i8",
                "int16": "i16",
                "int32": "i32",
                "int64": "i64",
            }

def short_dtype(x: Union[np.ndarray, np.generic]):
    return dtnames.get(x.dtype.name, str(x.dtype))

# %% ../nbs/00_repr_str.ipynb 9
def lovely( x       :Union[np.ndarray, np.generic], # The data you want to explore
            plain   :bool   =False,                 # Plain old way
            verbose :bool   =False,                 # Both summaty and plain
            depth   :int    =0,                     # Show deeper summary, up to `depth`
            lvl     :int    =0,                     # Indentation level
            color   :O[bool]=None                   # Override `get_config().color`
            ) -> str:                               # The summary

    "Pretty-print the stats of a numpy array or scalar"

    if plain or not isinstance(x, (np.ndarray, np.generic)) or np.iscomplexobj(x) or not np.issubdtype(x.dtype, np.number):
        return repr(x)

    conf = get_config()

    if isinstance(x, np.generic):
        tname = None
    else:
        tname = "array" if type(x) == np.ndarray else type(x).__name__.split(".")[-1]

    shape = str(list(x.shape)) if x.ndim else None
    type_str = sparse_join([tname, shape], sep="")

    color = get_config().color if color is None else color
    if in_debugger(): color = False

    numel = None
    if x.shape and max(x.shape) != x.size:
        numel = f"n={x.size}"
        if get_config().show_mem_above <= x.nbytes:
            numel = sparse_join([numel, f"({bytes_to_human(x.nbytes)})"])
    elif get_config().show_mem_above <= x.nbytes:
        numel = bytes_to_human(x.nbytes)

    common = np_to_str_common(x, color=color)
    dtype = short_dtype(x)

    vals = pretty_str(x) if 0 < x.size <= 10 else None
    res = sparse_join([type_str, dtype, numel, common, vals])

    if verbose:
        res += "\n" + repr(x)

    if depth and x.ndim > 1:
        deep_width = min(x.shape[0], conf.deeper_width) # Print at most this many lines
        with config(show_mem_above=np.inf):
            deep_lines = [ " "*conf.indent*(lvl+1) + lovely(x[i,:], depth=depth-1, lvl=lvl+1)
                                for i in range(deep_width)]

            # If we were limited by width, print ...
            if deep_width < x.shape[0]: deep_lines.append(" "*conf.indent*(lvl+1) + "...")

            res += "\n" + "\n".join(deep_lines)

    return res
