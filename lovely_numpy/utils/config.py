# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03d_utils.config.ipynb.

# %% auto 0
__all__ = ['set_config', 'get_config', 'config']

# %% ../../nbs/03d_utils.config.ipynb 3
from copy import copy
from types import SimpleNamespace
from typing import Callable, Union, Optional, TypeVar
from contextlib import contextmanager

import numpy as np

# %% ../../nbs/03d_utils.config.ipynb 4
class Config(SimpleNamespace):
    "Config"
    def __init__(self,
            precision     = 3,    # Digits after `.`
            threshold_max = 3,    # .abs() larger than 1e3 -> Sci mode
            threshold_min = -4,   # .abs() smaller that 1e-4 -> Sci mode
            sci_mode      = None, # Sci mode (2.3e4). None=auto
            indent        = 2,    # Indent for .deeper()
            color         = True, # ANSI colors in text
            repr          = None, # Use func e.g. `lovely` for `repr(np.ndarray)`
            str           = None, # Use func e.g. `lovely` for `str(np.ndarray)`
            plt_seed      = 42,   # Sampling seed for `plot`
            fig_close     = True, # Close matplotlib Figure
    ):
        super().__init__(**{k:v for k,v in locals().items() if k not in ["self", "__class__"]})

_defaults = Config()

_config = copy(_defaults)

# %% ../../nbs/03d_utils.config.ipynb 7
# Allows passing None as an argument to reset the 
class _Default():
    def __repr__(self):
        return "Ignore"
D = _Default()
Default = TypeVar("Default")

# %% ../../nbs/03d_utils.config.ipynb 8
def set_config( precision       :Optional[Union[Default,int]]     =D,
                threshold_min   :Optional[Union[Default,int]]     =D,
                threshold_max   :Optional[Union[Default,int]]     =D,
                sci_mode        :Optional[Union[Default,bool]]    =D,
                indent          :Optional[Union[Default,bool]]    =D,
                color           :Optional[Union[Default,bool]]    =D,
                repr            :Optional[Union[Default,Callable]]=D,
                str             :Optional[Union[Default,Callable]]=D,
                plt_seed        :Optional[Union[Default,int]]     =D,
                fig_close       :Optional[Union[Default,bool]]    =D,
                ) -> None:

    "Set config variables"
    args = locals().copy()
    for k,v in args.items():
        if v != D:
            
            # set_config(repr=func)             -> Set both `repr` and `str`.
            # set_config(repr=func, str=None)   -> Set `repr`, unset `str``
            # set_config(str=func)              -> Set `str`` only, don't tuch `repr``
            # set_config(repr=None)             -> Unset `repr`and `str``
            # set_config(str=None)              -> Unset `str` only

            if k == "repr":
                np.set_string_function(v, True)
                if args["str"] == D:
                    np.set_string_function(v, False)
            if k == "str":
                np.set_string_function(v, False)

            if v is None:
                setattr(_config, k, getattr(_defaults, k))
            else:
                setattr(_config, k, v)

# %% ../../nbs/03d_utils.config.ipynb 9
def get_config():
    "Get a copy of config variables"
    return copy(_config)

# %% ../../nbs/03d_utils.config.ipynb 10
@contextmanager
def config( precision       :Optional[Union[Default,int]]     =D,
            threshold_min   :Optional[Union[Default,int]]     =D,
            threshold_max   :Optional[Union[Default,int]]     =D,
            sci_mode        :Optional[Union[Default,bool]]    =D,
            indent          :Optional[Union[Default,bool]]    =D,
            color           :Optional[Union[Default,bool]]    =D,
            repr            :Optional[Union[Default,Callable]]=D,
            str             :Optional[Union[Default,Callable]]=D,
            plt_seed        :Optional[Union[Default,int]]     =D,
            fig_close       :Optional[Union[Default,bool]]    =D,
            ):
    "Context manager for temporarily setting config options"
    new_opts = { k:v for k, v in locals().items() if v != D}
    old_opts = copy(get_config().__dict__)


    try:
        set_config(**new_opts)
        yield
    finally:
        set_config(**old_opts)
