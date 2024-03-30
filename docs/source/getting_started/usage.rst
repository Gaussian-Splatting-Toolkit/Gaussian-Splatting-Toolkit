Usage
=====

The Gaussian Splatting Toolkit provides various commands and functionalities for 3D reconstruction and new view synthesis. Below are some common usage scenarios:

Download Opensource Datasets
-----------------------------

To download opensource datasets for testing and experimentation:

.. code-block:: bash

   gs-download-data gstk --save-dir /path/to/save/dir --capture-name all

Data Processing
---------------

Process your data for use with the toolkit:

**Extract from video:**

.. code-block:: bash

   gs-process-data video --data /path/to/video --output-dir /path/to/output-dir --num-frames-target 1000

**Extract from images:**

.. code-block:: bash

   gs-process-data images --data /path/to/rgb/folder --output-dir /path/to/output-dir

**Extract with both RGB and depth:**

.. code-block:: bash

   gs-process-data images --data /path/to/rgb/folder --depth-data /path/to/depth/folder --output-dir /path/to/output-dir

Train the Gaussian Splatting
----------------------------

Train the Gaussian Splatting model on your processed data:

.. code-block:: bash

   gs-train gaussian-splatting --data /path/to/processed/data

Visualize the Results
---------------------

Visualize the results using the viewer:

.. code-block:: bash

   gs-viewer --load-config outputs/path/to/config.yml

Render RGB and Depth
--------------------

Render RGB and depth images from a specified trajectory or camera pose:

**From trajectory:**

.. code-block:: bash

   gs-render trajectory --trajectory-path /path/to/trajectory.json --config-file /path/to/ckpt/config.yml

**From camera pose:**

.. code-block:: bash

   gs-render pose --config-file /path/to/config.yml --output-dir /path/to/output/folder/

Exporting Results
-----------------

Export various results such as gaussians, camera poses, point cloud, and TSDF:

**Export gaussians as PLY:**

.. code-block:: bash

   gs-export gaussian-splat --load-config /path/to/config.yml --output-dir exports/gaussians/

**Export camera poses:**

.. code-block:: bash

   gs-export camera-poses --load-config /path/to/config.yml --output-dir exports/cameras/

**Export point cloud:**

.. code-block:: bash

   gs-export point-cloud --load-config /path/to/config.yml --output-dir exports/pcd/

**Export TSDF:**

.. code-block:: bash

   gs-export offline-tsdf --load-config /path/to/config.yml --output-dir exports/tsdf/

With mask:

.. code-block:: bash

   gs-export offline-tsdf --render-path /path/to/rendered/folder --output-dir exports/tsdf/ --mask-path /path/to/mask

Using prompt:

.. code-block:: bash

   gs-export offline-tsdf --render-path /path/to/rendered/folder --output-dir exports/tsdf/ --seg-prompt your.prompt

These are the basic usage scenarios for the Gaussian Splatting Toolkit. For more advanced features and detailed documentation, please refer to the cli.
