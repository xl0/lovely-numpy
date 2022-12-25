# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_repr_chans.ipynb.

# %% auto 0
__all__ = ['chans']

# %% ../nbs/05_repr_chans.ipynb 3
from typing import Any, Optional as O
from functools import cached_property

import numpy as np
from matplotlib import axes, figure, pyplot as plt
from IPython.core.pylabtools import print_figure

from .repr_rgb import fig_rgb, rgb
from .utils.colormap import InfCmap, get_cmap
from .utils.config import config, get_config

# %% ../nbs/05_repr_chans.ipynb 4
def fig_chans(  x           :np.ndarray,      # Input array
                cmap        :str="twilight",  # Use matplotlib colormap by this name
                cm_below    :str="blue",
                cm_above    :str="red",
                cm_ninf     :str="cyan",
                cm_pinf     :str="fuchsia",
                cm_nan      :str="yellow",
                gutter_px   :int=3,         # Draw write gutters when tiling the images
                frame_px    :int=1,         # Draw black frame around each image
                scale       :int=1,         # Stretch the image. Only itegers please.
                cl          :Any=True,
                view_width  :int=966,
                ax          :O[axes.Axes]=None
        ) -> figure.Figure:
    """
    Process individual channels of a ndarray that can be interpreted as as image
    """
    
    assert x.ndim >= 2, f"Expected a 2+ dim input, got {x.shape}={x.ndim}"
    if x.ndim == 2: x = x[None]
    
    if cl: # Convert to [..., C, H, W].
        x = np.swapaxes(np.swapaxes(x, -2, -1), -3, -2)

    ### XXX Do we want a way to pass a custom cmap instead of mpl one?
    inf_cmap = InfCmap(cmap=get_cmap(cmap),
                  below=cm_below, above=cm_above,
                  nan=cm_nan, ninf=cm_ninf, pinf=cm_pinf)

    return fig_rgb(inf_cmap(x), cl=True, gutter_px=gutter_px, frame_px=frame_px, scale=scale, view_width=view_width, ax=ax)


# %% ../nbs/05_repr_chans.ipynb 5
class ChanProxy():   
    def __init__(self, x: np.ndarray):
        self.x = x
        self.params = dict( cmap        ="twilight", 
                            cm_below    ="blue",
                            cm_above    ="red",
                            cm_ninf     ="cyan",
                            cm_pinf     ="fuchsia",
                            cm_nan      ="yellow",
                            view_width  =966,
                            gutter_px   =3,
                            frame_px    =1,
                            scale       =1,
                            cl          =True,
                            ax          =None)

    def __call__(self,
                 cmap       :O[str] =None, 
                 cm_below   :O[str] =None,
                 cm_above   :O[str] =None,
                 cm_ninf    :O[str] =None,
                 cm_pinf    :O[str] =None,
                 cm_nan     :O[str] =None,
                 view_width :O[int] =None,
                 gutter_px  :O[int] =None,
                 frame_px   :O[int] =None,
                 scale      :O[int] =None,
                 cl         :Any    =None,
                 ax         :O[axes.Axes]=None):
        
        self.params.update( {   k:v for
                                k,v in locals().items()
                                if k != "self" and v is not None } )
        _ = self.fig # Trigger figure generation
        return self

    @cached_property
    def fig(self) -> figure.Figure:
        return fig_chans(self.x, **self.params)

    def _repr_png_(self):
        return print_figure(self.fig, fmt="png", pad_inches=0,
            metadata={"Software": "Matplotlib, https://matplotlib.org/"})


# %% ../nbs/05_repr_chans.ipynb 6
def chans(  x           :np.ndarray,      # Input array
            cmap        :str="twilight",  # Use matplotlib colormap by this name
            cm_below    :str="blue",
            cm_above    :str="red",
            cm_ninf     :str="cyan",
            cm_pinf     :str="fuchsia",
            cm_nan      :str="yellow",
            gutter_px   :int=3,         # Draw write gutters when tiling the images
            frame_px    :int=1,         # Draw black frame around each image
            scale       :int=1,         # Stretch the image. Only itegers please.
            cl          :Any=True,
            view_width  :int=966,
            ax          :O[axes.Axes]=None
        ) -> ChanProxy:
    "Map x values to colors. RGB[A] color is added as channel-last"
    args = locals()
    del args["x"]

    return ChanProxy(x)(**args)
