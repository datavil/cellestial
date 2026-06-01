I need a re-write for benchmarks,

I deleted it all.

Every case should be its own script under benchmarks/scripts/*.py
I should be able to run it via CLI.
every script should accept "-r" for replica number "-d" for path to dataset.

for example, benchmarks/scripts/heatmap.py

should import libraries and load dataset.

and have two functions _cl_heatmap() which uses cellestial and _sc_heatmap() which uses scanpy pl.

for a fair comparison cellestial should use tooltips="none" and sampling="none".

Both libraries should render to svg (use cl.save() for cellestial figures to be disgarged after render)
Both functions' peak memory use and performance should be measured.

every output to be appended to a benchmarks/results.feather, 

with columns names

library,case,replica_id,dataset,n_obs,n_cols,n_items,memory(MB),time(s)

where,
library: is one of cellestial or scanpy
case: is heatmap, umap etc.
replica_id: 1, 2, 3 etc
dataset: "atlas200k" for data/atlas200k.h5ad and "pbmc3k_pped" for data/pbmc3k_pped.h5ad (like str.split("/")[-1].split(".")[0].split("_")[-1])
n_obs: number of observations
n_cols: number of columns actively used for the plot.
n_items: number of items (n_obs*n_cols)
memory(s): peak memory usage in MB
time(s): time in seconds

the benchmarks should be ran indepently and modularly.

so that when I run `heatmap` benchmarks, it should append to results.feather, (creates it if it does not exist)

if the same exact "library,case,replica_id,dataset," already exists new coming results MUST overwrite it.


we are to test following functions.

umap, one with a variable key (gene) and one with categorical key
- umap_cat.py
- umap_var.py
(we must do it in separate scripts as to prevent scanpy caching)

heatmap,
- heatmap.py

it will need -n flag for number of genes to use 
which should vary from 5,20,100,200

spatial, one with a variable key (gene) and one with categorical key
- spatial_cat.py
- spatial_var.py

boxplot, 
- boxplot.py

Also, I need you to a write a CLI.sh to run these all modularly (or source it all).


pbmc3k and atlas200k for heatmap, umap, boxplot,

data/V1_Human_Lymph_Node and data/V1_Mouse_Kidney for spatial.

Make a detailed plan, clarify steps and uncertainities. 

A share your plan with me, do not make any changes yet.

Ask me your question to clarify.