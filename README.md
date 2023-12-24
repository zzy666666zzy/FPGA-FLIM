# FPGA-FLIM

This repo includes code synthesizing fluorescence lifetime imaging (FLIM) data, deep learning architecture training and inference lifetime, and hardware code for FPGA. The code is provided for the paper "Compact and robust deep learning architecture for fluorescence lifetime imaging and FPGA implementation" to re-produce or generate new ideas based on this work.

"This paper reports a bespoke adder-based deep learning network for time-domain fluorescence
lifetime imaging (FLIM). By leveraging the l1-norm extraction method, we propose a 1D Fluorescence
Lifetime AdderNet (FLAN)without multiplication-based convolutions to reduce the computational
complexity. Further, we compressed fluorescence decays in temporal dimension using a log-scale
merging technique to discard redundant temporal information derived as log-scaling FLAN (FLAN
+LS). FLAN+LS achieves 0.11 and 0.23 compression ratios compared with FLAN and a conventional
1D convolutional neural network (1D CNN)while maintaining high accuracy in retrieving lifetimes.
We extensively evaluated FLAN and FLAN+LS using synthetic and real data. A traditional fitting
method and other non-fitting, high-accuracy algorithms were compared with our networks for
synthetic data. Our networks attained a minor reconstruction error in different photon-count
scenarios. For real data, we used fluorescent beads’ data acquired by a confocal microscope to validate
the effectiveness of real fluorophores, and our networks can differentiate beads with different
lifetimes. Additionally, we implemented the network architecture on a field-programmable gate array
(FPGA)with a post-quantization technique to shorten the bit-width, thereby improving computing
efficiency. FLAN+LS on hardware achieves the highest computing efficiency compared to 1D CNN
and FLAN. We also discussed the applicability of our network and hardware architecture for other
time-resolved biomedical applications using photon-efficient, time-resolved sensors."

*FLIM data generation is in `FLIM_DATA_GEN` folder  
*Training and inference code is in `DL_FLIM` folder  
*Hardware architecture (high-level synthesis and bare-metal ARM codes) is in `HW_FPGA` folder  

This code was developed using Matlab, PyTorch, and Vivado (and HLS and Vivado SDK) 2018.2,

FPGA-based on-chip fluorescence lifetime analysis.

Citation:
Zang, Zhenya, Dong Xiao, Quan Wang, Ziao Jiao, Yu Chen, and David Day Uei Li. "Compact and robust deep learning architecture for fluorescence lifetime imaging and FPGA implementation." Methods and Applications in Fluorescence 11, no. 2 (2023): 025002.
