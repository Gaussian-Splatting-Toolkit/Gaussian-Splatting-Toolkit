Command-Line Interface (CLI)
============================

The Gaussian Splatting Toolkit provides a CLI for easy interaction with its functionalities. Below are the main commands and their descriptions:

Download Opensource Datasets
-----------------------------

**Command:**

.. code-block:: bash

   gs-download-data

**Description:**

Downloads opensource datasets for use with the toolkit.

**Usage:**

.. code-block:: bash

   gs-download-data gstk --save-dir /path/to/save/dir --capture-name all

Data Processing
---------------

**Command:**

.. code-block:: bash

   gs-process-data

**Description:**

Processes data for use with the Gaussian Splatting Toolkit.

**Usage:**

Extract from video:

.. code-block:: bash

   gs-process-data video --data /path/to/video --output-dir /path/to/output-dir --num-frames-target 1000

Extract from images:

.. code-block:: bash

   gs-process-data images --data /path/to/rgb/folder --output-dir /path/to/output-dir

Train the Gaussian Splatting
----------------------------

**Command:**

.. code-block:: bash

   gs-train

**Description:**

Trains the Gaussian Splatting model on the processed data.

**Usage:**

.. code-block:: bash

   gs-train gaussian-splatting --data /path/to/processed/data

Visualize the Results
---------------------

**Command:**

.. code-block:: bash

   gs-viewer

**Description:**

Visualizes the results using the viewer.

**Usage:**

.. code-block:: bash

   gs-viewer --load-config outputs/path/to/config.yml

Render RGB and Depth
--------------------

**Command:**

.. code-block:: bash

   gs-render

**Description:**

Renders RGB and depth images from a specified trajectory or camera pose.

**Usage:**

From trajectory:

.. code-block:: bash

   gs-render trajectory --trajectory-path /path/to/trajectory.json --config-file /path/to/ckpt/config.yml

From camera pose:

.. code-block:: bash

   gs-render pose --config-file /path/to/config.yml --output-dir /path/to/output/folder/

Exporting Results
-----------------

**Command:**

.. code-block:: bash

   gs-export

**Description:**

Exports various results such as gaussians, camera poses, point cloud, and TSDF.

**Usage:**

Export gaussians as PLY:

.. code-block:: bash

   gs-export gaussian-splat --load-config /path/to/config.yml --output-dir exports/gaussians/

Export camera poses:

.. code-block:: bash

   gs-export camera-poses --load-config /path/to/config.yml --output-dir exports/cameras/

Export point cloud:

.. code-block:: bash

   gs-export point-cloud --load-config /path/to/config.yml --output-dir exports/pcd/

Export TSDF with mask:

.. code-block:: bash

   gs-export offline-tsdf --render-path /path/to/rendered/folder --output-dir exports/tsdf/ --mask-path /path/to/mask

These are the main CLI commands provided by the Gaussian Splatting Toolkit. For more detailed information on each command and its options, please refer to the toolkit's documentation.
