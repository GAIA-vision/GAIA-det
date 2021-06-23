GAIA-det
^^^^^^
**The code is almost complete, but the documentation haven't been completed yet. Please stay tuned.**

Introduction 
------------
GAIA-det is a open source object detection that helps you with your customized AI solutions. It is built on top of gaiavision_ and mmdet_.


.. _gaiavision: https://github.com/GAIA-vision/GAIA-cv
.. _mmdet: https://github.com/open-mmlab/mmdetection

It provides functionalities that help the customization of AI solutions.

- Design customized search space of any type with little efforts.
- Manage models in search space according to your rules.
- Integrate datasets of various sources.


Installation
------------

- Install gaiavision_.
- Install mmdet_.
- Install ``gaiadet``:

.. code-block:: bash
  
  git clone https://github.com/GAIA-vision/GAIA-det . && cd GAIA-det
  pip install -r requirements.txt
  pip install -e .



Data Preparation
----------------

Please refer to DATA_PREPARATION_.

.. _DATA_PREPARATION: https://github.com/GAIA-vision/GAIA-det/blob/dev/docs/DATA_PREPARATION.rst

Usage
-----
Please refer to USAGE_.

.. _USAGE: https://github.com/GAIA-vision/GAIA-det/blob/dev/docs/USAGE.rst

Acknowledgements
---------------

- This repo is constructed on top of mmdet_.




