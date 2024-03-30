Quick Start
===========

Once you have installed the Gaussian Splatting Toolkit, you can quickly get started with a basic example:

1. Download Sample Data
-----------------------

Use the provided command to download some open-source datasets for testing:

.. code-block:: bash

   gs-download-data gstk --save-dir /path/to/save/dir --capture-name all

2. Process the Data
-------------------

Process the downloaded data for use with the Gaussian Splatting Toolkit:

.. code-block:: bash

   gs-process-data images --data /path/to/rgb/folder --output-dir /path/to/output-dir

3. Train the Gaussian Splatting Model
-------------------------------------

Train a model using the processed data:

.. code-block:: bash

   gs-train gaussian-splatting --data /path/to/processed/data

4. Visualize the Results
------------------------

After training, you can visualize the results using the provided viewer:

.. code-block:: bash

   gs-viewer --load-config outputs/path/to/config.yml

5. Render RGB and Depth Images
------------------------------

You can also render RGB and depth images from a specific trajectory:

.. code-block:: bash

   gs-render trajectory --trajectory-path /path/to/trajectory.json --config-file /path/to/ckpt/config.yml

This quick start guide should help you get up and running with the Gaussian Splatting Toolkit. For more detailed usage and advanced features, please refer to the full documentation.
