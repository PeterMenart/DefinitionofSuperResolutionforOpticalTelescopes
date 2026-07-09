# DefinitionofSuperResolutionforOpticalTelescopes
Code repository for the paper Definition of Multi-Parameter Super-Resolution for Optical Telescopes. We provide a Fisher information framework for analyzing imaging system resolution. Specifically,
the Cramer-Rao lower bound (CRB), calculated from the Fisher information, specifies the minimum error with which our parameters of interest can be specified. In this code, we calculate the classical
CRB for direct imaging, which serves as a benchmark for super-resolution, and the quantum CRB, which provides the ultimate error bound for any measurement system. We do the calculations for the case of imaging N point sources, resulting in 3N parameters to estimate - the y position, x position, and brightness of each point source.

We also provide a natural gradient descent algorithm that achieves the classical CRB for direct imaging and can be adapted to super-resolution techniques using Pytorch's autograd functionality.

### Conventions

### Script 1

### Script 2












