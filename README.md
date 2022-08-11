# PyTorch Implementation of Alchemical Kernels
 
Reproduced from [Feature optimization for atomistic machine learning yields a data-driven construction of the periodic table of the elements](https://doi.org/10.1039/C8CP05921G) with a simplified loss function compared to the original paper. The idea is to use kernel ridge regression to fit the structure-energy relationship from the SOAP descriptor representation. Nevertheless, to reduce the dimensionality of the kernel, an alchemical kernel is used, which describes the elemental information of the 39 elements present in the dataset as a linear combination of 4 pseudo-elements, or alchemical elements. This then allows us to produce a reconstruction of the periodic table of elements as shown in the figure below, for 2, 3, and 4 pseudo-elements (taken from the original paper).
<p align="center">
<img src="https://user-images.githubusercontent.com/68380659/184111486-a273b817-bd64-4e75-88f0-ad59a5ea3b69.gif" alt="periodic_table" width="50%"/>
</p>
The dataset used is an elpasolite dataset of 8k structure from <a href="https://doi.org/10.1103/PhysRevLett.117.135502">Machine Learning Energies of 2 Million Elpasolite (<i>ABC<sub>2</sub>D<sub>6</sub></i>) Crystals</a>. My slightly different approach in terms of learning the coupling parameters, compared to the original paper, is given in the figure below.

![model_structure](https://user-images.githubusercontent.com/68380659/184113100-99b45fcc-8244-4e1a-be68-7eff9ff61a04.png)


We use two different datasets during training: a) <b>training dataset</b>: learns the weights that describe the kernel-energy relationship through linear matrix regression and b) <b>optimization dataset</b>: learns the coupling parameters <i>U</i> that transfer the full SOAP descriptor into the reduced SOAP vector with the reduced dimensionality. We train the weights and coupling parameters simultaneously. 

It was found that there is training instability if we detach the gradients in the training dataset path, betwen the weights and the <i>U</i> coupling parameters. Hence the parameter updates of the <i>U</i> matrix are updated using gradients that propagate through both paths - training and optimization.
