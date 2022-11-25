# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_repr_chans.ipynb.

# %% auto 0
__all__ = ['chans']

# %% ../nbs/05_repr_chans.ipynb 3
import numpy as np
from . import lovely

from .repr_rgb import rgb
from .utils.colormap import InfCmap, get_cmap

# %% ../nbs/05_repr_chans.ipynb 4
def chans(  t: np.ndarray,      # Input tensor 
            cmap = "coolwarm",  # Use matplotlib colormap by this name
            cm_below="blue", cm_above="red",
            cm_ninf="cyan", cm_pinf="fuchsia",
            cm_nan="yellow",
            gutter_px=3,   # Draw write gutters when tiling the images
            frame_px=1,    # Draw black frame around each image
            view_width=966):    
    """
    Process individual channels of a tensor that can be interpreted as as image
    `x` and `y` specify which dimensions should be used as spatial ones.
    """
    
    assert t.ndim >= 2, f"Expected a 2 or 3-dim input, got {t.shape}={t.ndim}"
    if t.ndim == 2: t = t[None]
    
    ### XXX Do we want a way to pass a custom cmap instead of mpl one?
    inf_cmap = InfCmap(cmap=get_cmap(cmap),
                  below=cm_below, above=cm_above,
                  nan=cm_nan, ninf=cm_ninf, pinf=cm_pinf)

    return rgb(inf_cmap(t), cl=True, gutter_px=gutter_px, frame_px=frame_px, view_width=view_width)


