#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lets_plot import *
LetsPlot.setup_html()

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

p1 = (
    cl.plot(data, aes(x="cell_type_lvl1", y="n_genes"))
)
p1 # plot object without layers


# In[2]:


from lets_plot import *
LetsPlot.setup_html()

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

p2 = (
    cl.plot(data, aes(x="cell_type_lvl1", y="n_genes"))
    + geom_violin(aes(fill="cell_type_lvl1"), scale="width")
    + geom_boxplot(width=0.2,outlier_size=0)
    + scale_fill_viridis()
)
p2

