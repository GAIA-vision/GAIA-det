Usage
-----
GAIA-dets provides the following tools:

- `Train supernet`_: Build up a search space and train a supernet on large amount of data.

- `Test supernet`_: Define sampling rules and test subnets sampled from supernet based on rules.

- `Finetune supernet`_: Define sampling rules and finetune subnets sampled from supernet based on rules.

- `Count flops`_: Count flops of subnets in search space.

- `Extract subnets`_: Extract weights of subnets from supernets.

.. _`Train supernet`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#training-supernet
.. _`Test supernet`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#tsastask-specific-architecture-selection
.. _`Finetune supernet`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#tsastask-specific-architecture-selection
.. _`Count flops`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#count-flops
.. _`Extract subnets`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#extract-weights-and-finetune-with-your-own-tricks



Workflow
--------
To begin with, you need to get a powerful supernet to do all things:

- You can build up your own search space and `train supernet`_ on yourself.

- Or, you can use arch_ and ckpt(coming soon) of supernet in this repo.

.. _`train supernet`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#training-supernet
.. _arch: https://github.com/GAIA-vision/GAIA-det/blob/master/configs/_dynamic_/models/faster_rcnn_fpn_ar50to101_gsync.py

Then you need to maintain overhead of subnets:

- You can `count flops`_ and params of subnets from your customized search space.

- You can use flops.json that records infomation of arch_ in this repo.

.. _`count flops`: https://github.com/GAIA-vision/GAIA-det/blob/master/docs/USAGE.rst#count-flops

During downstream customization:

- You could design `rules`_ and directly sample subnets for fast-finetuning. This would generate a file that records performance of subnets. Direct finetuning is usually applied when downstream label space is a subset of upstream label space.
- You could also design `rules`_ and directly sample subnets for testing. This could shrink the search space and you could apply fast-finetuning based on rules like this_. 

.. _`this`: https://github.com/GAIA-vision/GAIA-det/blob/master/configs/_dynamic_/rules/ar50to101_ft2e_rules.py
.. _`rules`: https://github.com/GAIA-vision/GAIA-det/blob/master/configs/_dynamic_/rules/140GF_rules.py

Final model training and extraction:

- You could directly use the best-performed fast-finetuned model.

- Or, you could extract weights of the best-performed arch from supernet, and finetune it with your own tricks(like DCN, Cascaded RCNN and etc.) 

Examples
--------

Train supernet 
>>>>>>>>>>>>>>>>>

.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/train_local.sh 8 configs/local_examples/train_supernet/faster_rcnn_ar50to101_gsync.py /path/to/work_dir

Count flops
>>>>>>>>>>>>>

.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/count_flops_local.sh 8 configs/local_examples/train_supernet/faster_rcnn_ar50to101_flops.py /path/to/work_dir

After this, you may have a ``/path/to/work_dir/flops.json`` which records flops of each model. It looks like this:

.. code-block:: json

  {"overhead": {"flops": 46314197434.0, "params": 28010543}, "arch": {"backbone": {"stem": {"width": 32}, "body": {"width": [48, 96, 192, 384], "depth": [2, 2, 5, 2]}}}, "data": {"input_shape": 480}}
  {"overhead": {"flops": 47443272634.0, "params": 33024047}, "arch": {"backbone": {"stem": {"width": 32}, "body": {"width": [48, 96, 192, 384], "depth": [2, 2, 5, 4]}}}, "data": {"input_shape": 480}}
  ...

TSAS(Task specific architecture selection): 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/finetune_local.sh $NUM_GPUS $CONFIG $WORK_DIR $SUPERNET_CKPT $FLOPS_JSON
  
Or, you could test subnets to shrink the search space:

.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/test_local.sh $NUM_GPUS $CONFIG $WORK_DIR $SUPERNET_CKPT $FLOPS_JSON
 
and then apply fast-finetune, with ``TEST_JSON=/path/to/work_dir/test_supernet/metric.json``:

.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/finetune_local.sh $NUM_GPUS $CONFIG $WORK_DIR $SUPERNET_CKPT $TEST_JSON
  
Extract weights and finetune with your own tricks
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
.. code-block:: bash

  cd /path/to/GAIA-det
  sh scripts/extract_subnet.sh $NUM_GPUS $CONFIG $WORK_DIR $SUPERNET_CKPT 

TSDS(Task specific data selection)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Coming soon.
