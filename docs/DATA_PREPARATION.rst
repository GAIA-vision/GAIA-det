Data preparation
^^^^^^^^^^^^^^^^

Upstream datasets:
-----------------
We recommend three datasets of object detection to train a supernet.

- COCO dataset
- Objects365 dataset
- OpenImages dataset


Prepare COCO dataset
>>>>>>>>>>>>>>>>>>>>

- Download the whole dataset from `link <https://cocodataset.org/#download>`__
- Formulate the data directory as follows:

.. code-block:: text

    /path/to/coco/
    ├─annotations/
    │ ├─instances_train2017.json
    │ ├─instances_val2017.json
    │ └─...
    └─images/
      │─train2017/
      │ ├─{image_id}.jpg
      │ └─...
      └─val2017/
        │─{image_id}.jpg
        └─...


Prepare Objects365 datset
>>>>>>>>>>>>>>>>>>>>>>>>>

- Download the whole dataset from `link <https://www.objects365.org/download.html>`__.
- Note that Objects365v1 which is the version we use in paper_ is not available now. Use v2 instead which holds more data.
- Formulate the data directory as follows:

.. _paper: https://arxiv.org/abs/2106.11346

.. code-block:: text

    /path/to/objects365/
    ├─annotations/
    │ ├─objects365_train.json
    │ ├─objects365_val.json
    │ └─...
    └─images/
      │─train/
      │ ├─{image_id}.jpg
      │ └─...
      │─val/
      │ ├─{image_id}.jpg
      │ └─...
      └─test/
        │─{image_id}.jpg
        └─...
        
- Convert metafile from coco-style to custom-style (optional):      

.. code-block:: bash

    cd /path/to/GAIA-det
    python tools/convert_datasets/coco2custom.py --data_dir /path/to/Objects365 --src_name objects365_train.json --dst_name objects365_generic_train.json
  
Prepare OpenImages datset
>>>>>>>>>>>>>>>>>>>>>>>>>

- Download the whole dataset from `link <https://storage.googleapis.com/openimages/web/download.html>`__.
- Note that the version we use in paper_ is OpenImages2019 Challenge, annotations could from `link <https://storage.googleapis.com/openimages/web/challenge2019_downloads.html>`__.
- Formulate the data directory as follows:

.. code-block:: text

    /path/to/OpenImages/
    ├─annotations/
    │ ├─bbox_labels_500_hierarchy.json
    │ ├─challenge-2019-classes-description-500.csv
    │ ├─challenge-2019-train-detection-bbox.csv
    │ └─...
    └─images/
      │─train/
      │ ├─{image_id}.jpg
      │ └─...
      │─val/
      │ ├─{image_id}.jpg
      │ └─...
      └─test-challenge/
        │─{image_id}.jpg
        └─...
        
- Convert metafile from coco-style to custom-style (required):

.. code-block:: bash

    cd /path/to/GAIA-det
    python tools/convert_datasets/oid2custom.py --oid_dir /path/to/OpenImages --dst_name oid500_generic_train.json
