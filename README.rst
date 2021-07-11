GAIA-det
^^^^^^
**Model and demo coming soon! All things would be ready on July 20th. Stay tuned.**

Introduction 
------------

GAIA-det is a open source object detection toolbox that helps you with your customized AI solutions. It is built on top of gaiavision_ and mmdet_. 
This repo includes a re-implementation of our CVPR2021 paper: `GAIA: A Transfer Learning System of Object Detection that Fits Your Needs <https://arxiv.org/abs/2106.11346>`__.


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

Prepare Supernet
-----------------

+-------------+------------+------------+--------------------------------------------------------------------+-------------+
| Type        | Backbone   | Model      | Cloud Storage                                                      | Password    | 
+=============+============+============+====================================================================+=============+
| WEIGHTS     | AR50to101  | Faster     |  `BaiduCloud <https://pan.baidu.com/s/1V0H02yjssQKYBYF5lu_6Gw>`__  | ``tm5n``    | 
+-------------+------------+------------+--------------------------------------------------------------------+-------------+
| FLOPS_LUT   | AR50to101  | Faster     |  `BaiduCloud <https://pan.baidu.com/s/18kYu6pC0JdGyGYdK9HkC8A>`__  | ``ttwq``    | 
+-------------+------------+------------+--------------------------------------------------------------------+-------------+


Benchmark
----------

Finetuning(Upstream-COCO)
~~~~~~~~~~~~~~~~~~~~~~~~~

+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| Backbone   | Model      | Depth         | Width                | Input       | Lr        | GFLOPS     |  box AP          |  box AP              |
|            |            |               |                      | Scale       | schd      |            |  (paper)         |  (this repo)         |
+============+============+===============+======================+=============+===========+============+==================+======================+
| ResNet50   | Faster     | 3, 4, 6, 3    |64, 64, 128, 256, 512 | 800         | 1x        | 139        |   37.1           |   37.6               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| ResNet50   | Faster     | 3, 4, 6, 3    |64, 64, 128, 256, 512 | 800         | 4x        | 139        |   None           |   40.3               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| 45-50GF    | Faster     | 2, 4, 5, 3    |64, 64, 96, 192, 384  | 480         | 1x        | 49         |   40.4           |   40.7               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| 70-75GF    | Faster     | 4, 6, 27, 4   |48, 64, 128, 192, 512 | 480         | 1x        | 71         |   42.6           |   43.1               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| 85-90GF    | Faster     | 3, 4, 21, 4   |48, 64, 160, 192, 640 | 560         | 1x        | 90         |   43.6           |   44.4               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| 110-115GF  | Faster     | 2, 4, 25, 4   |64, 64, 160, 192, 640 | 640         | 1x        | 115        |   44.5           |   44.8               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+
| 135-140GF  | Faster     | 4, 6, 19, 3   |48, 48, 128, 320, 512 | 640         | 1x        | 139        |   45.3           |   45.6               |
+------------+------------+---------------+----------------------+-------------+-----------+------------+------------------+----------------------+

Finetuning(Downstream-BDD100k)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comming soon.


Data Preparation
----------------

Please refer to DATA_PREPARATION_.

.. _DATA_PREPARATION: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/DATA_PREPARATION.rst

Usage
-----
Please refer to USAGE_ for generic use.

.. _USAGE: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst





