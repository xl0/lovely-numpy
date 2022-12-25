#!/usr/bin/env python
# %%
# %matplotlib ipympl

import numpy as np
import matplotlib.pyplot as plt

from lovely_numpy import lo, config, set_config

#%%

numbers = np.load("../nbs/mysteryman.npy").transpose(1,2,0)
in_stats = ( (0.485, 0.456, 0.406),     # mean
             (0.229, 0.224, 0.225) )    # std

mean = np.array(in_stats[0])
std = np.array(in_stats[1])
numbers = (numbers*std + mean).clip(0,1)


#%%

print("make figure, provide axes, plt.show()")

fig = plt.figure(figsize=(8,3))
fig.set_constrained_layout(True)
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1,1:])

ax2.set_axis_off()
ax3.set_axis_off()

lo(numbers).plt(ax=ax1)
lo(numbers).rgb(ax=ax2)
lo(numbers).chans(ax=ax3)

plt.show();

# %%

print("numers.plt(), plt.show() - nothing should happen")
lo(numbers).plt();
plt.show()

# %%

print("fig_close=False, numers.chans(), plt.show()")
set_config(fig_close=False)
lo(numbers).chans();
plt.show()

#%%
print("fig_show=True, numers.rgb(), no plt.show() - should still show")
set_config(fig_show=True)
lo(numbers).rgb();


# %%
