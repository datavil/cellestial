#!/usr/bin/env python
# coding: utf-8

# In[1]:


from lets_plot import *
LetsPlot.setup_html()

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

p1 = (
    cl.bar(data, mapping=aes('leiden', fill='predicted_doublet'))
    + scale_fill_brewer(palette='Set2', direction=-1)
)
p1


# In[2]:


from lets_plot import *
LetsPlot.setup_html()

import cellestial as cl
import scanpy as sc

data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

p2 = (
    cl.bar(data, mapping=aes('cell_type_lvl1', fill='leiden'))
    + scale_fill_brewer(palette='Set2')
)
p2

