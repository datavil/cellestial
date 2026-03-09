Performance & Scalability
=========================

**Cellestial** builds a `polars.DataFrame` from the single-cell data type (e.g. `AnnData`) to be passed into `ggplot()`, either directly or after necessary processing. 
The process of building the DataFrame mostly avoids using unnecessary columns allowing minimal memory usage and scalability. 
Leveraging the *blazing fast* nature of `Polars <https://pola.rs/>`_, **Cellestial** ensures performance.


Plotting, the core step, is handled by `Lets-Plot <https://lets-plot.org/python/>`_ which is fast and scalable.


Example: 100k cells
-------------------

Using `Human Breast Cancer Single Cell Atlas <https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb>`_.

.. image:: _static/breast_cancer_atlas_umap.png
   :width: 100%