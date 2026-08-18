# DefinitionofSuperResolutionforOpticalTelescopes
Code repository for the paper Definition of Multi-Parameter Super-Resolution for Optical Telescopes. We provide a Fisher information framework for analyzing imaging system resolution. Specifically,
the Cramer-Rao lower bound (CRB), calculated from the Fisher information, specifies the minimum error with which our parameters of interest can be specified. In this code, we calculate the classical
CRB for direct imaging, which serves as a benchmark for super-resolution, and the quantum CRB, which provides the ultimate error bound for any measurement system. We do the calculations for the case of imaging N point sources, resulting in 3N parameters to estimate - the y position, x position, and brightness of each point source.

We also provide a natural gradient descent algorithm that achieves the classical CRB for direct imaging and can be adapted to super-resolution techniques using Pytorch's autograd functionality.

# \*\*Code is available in the releases tab under the release MainCode**

### Conventions and Notes
* Variables that contain parameter values, such as `params_curr`, have shape (3, Number of sources), and the parameter values are ordered as (y position, x position, brightness). For example, the y position of the second source would be indexed as [0, 1].
* In these scripts, we mostly use analytical expressions for derivatives since they are not difficult to calculate for the Airy Disk model. For more complicated forward models, autograd functionality can be used to automatically calculate the derivatives. An example of this is given in the QFIM calculation.
* The code is designed to be run in sections as denoted by the #%% markings. If not using an IDE that supports this, you may try porting the code to a Jupyter notebook.
* We use Pytorch with CUDA to run the code on GPU, but most of the code should not be difficult to run on CPU. However, running the estimation algorithm for many trials may be quite slow on CPU.

### Scripts
Here we provide a brief description of each Python script.
##### UserPackage_AiryDiskModel.py
The class provided in this script is the most important part of the code. The class contains functions that calculate the classical Fisher information, quantum Fisher information, and helper functions for the natural gradient descent algorithm. Creating an instance of the class requires specifying the relevant parameters of the optical imaging system.

##### Figure2_CFI_QFI_Example.py
This script provides an example of calculating the classical and quantum error limits for various point source scenes, and will reproduce the results of Figure 2 of the paper.

##### Figure3_NGalgorithm_Example.py
This script provides an example of running the natural gradient descent algorithm on simulated noisy images and comparing the performance to the error limits. It can be used to reproduce the results of Figure 3 of the paper.

# Citing
If you use or reference this code for your own research, we ask you to cite the original paper:

Menart, P., Choi, S., Jacob, Z. "Definition of Multi-Parameter Super-Resolution for Optical Telescopes."











