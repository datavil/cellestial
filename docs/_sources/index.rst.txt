Cellestial |version|
====================

|pypi| |license| |polars| |letsplot|

The *grammar of graphics* for single-cell omics.


Installation
--------------
.. tab-set::

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install cellestial

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv add cellestial

   .. tab-item:: poetry
      :sync: poetry

      .. code-block:: bash

         poetry add cellestial

Documentation
--------

.. toctree::
   :maxdepth: 1

   API

.. toctree::
   :hidden:

   philosophy
   performance

About Lets-Plot
---------------

Cellestial is built on top of a powerful Python library, Lets-Plot. 
It is the best Python implementation of *ggplot2* with additional features such as **tooltips** and **zooming and panning**.
`Lets-Plot API <https://lets-plot.org/python/pages/api.html>`_

Example 
---------------
Hover over the plot 'geoms' to see tooltips, or use **toolbar** above the plot for **zooming and panning** options.

.. raw:: html
   :file: _static/overall.ggtb.html


.. |pypi| image:: https://img.shields.io/pypi/v/cellestial?color=377eb8
   :target: https://pypi.org/project/cellestial/
   :alt: PyPI version

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-ff0000
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License: Apache 2.0

.. |polars| image:: https://img.shields.io/badge/Powered%20by-Polars-377eb8?logo=polars&logoColor=white
   :target: https://www.pola.rs/
   :alt: Powered by Polars

.. |letsplot| image:: https://img.shields.io/badge/Graphics-Lets--Plot-FF00CC?logo=jetbrains&logoColor=white
   :target: https://lets-plot.org/
   :alt: Built with Lets-Plot