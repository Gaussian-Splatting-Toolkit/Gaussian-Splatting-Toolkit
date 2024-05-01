Installation
============

To install the Gaussian Splatting Toolkit, follow these steps:

1. Clone the Repository
-----------------------

Clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/H-tr/gaussian-splatting-toolkit.git --recursive

2. Navigate to the Toolkit Directory
------------------------------------

Change into the toolkit directory:

.. code-block:: bash

   cd gaussian-splatting-toolkit

3. Install Third-Party Dictionaries (if applicable)
---------------------------------------------------

* Colmap: Download the dictionary file from the Colmap repository

4. Install Dependencies
-----------------------

Install the required dependencies using pip:

.. code-block:: bash

   pip install -e .

Alternatively, you can use conda to create a virtual environment and install the dependencies:

.. code-block:: bash

   conda create -n gstk python=3.10.13 -y
   conda activate gstk
   pip install torch torchvision
   pip install -e .

Note: Replace the Python version and other dependencies as needed based on your project requirements.
