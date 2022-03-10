# Calculation of Metrics  for 3D Human Models



This repository contains a pytorch implementation for calculating  3D human model metrics.
In this repository, CUDA is used to accelerate the computation of metrics.
Compared to other repositories, we wrapped the computational code for CD, PSD, COS, L2, and IOU. 
An repository https://github.com/sunjc0306/3DRender  is required for the computation of COS and L2. 
An additional repository https://github.com/sunjc0306/sematicVoxel is required for the computation of IOU. 
The relevant computational formulas and characteristics of the metrics will be given in our paper.



## Requirements
- Python 3
- PyTorch
- numpy
- opendr


## Demo
Note: H_NORMALIZE and volume_res represent the height and resolution of the model, respectively. 
It is worth noting that the human model can be in any coordinate space, but its spatial symmetry must be guaranteed, e.g. [-0.5,-0.5,-0.5]* [0.5,0.5,0.5].
run the following script to calculate the error of the two models and obtain its quantitative evaluation.
```
python cal_metrics.py
```


## Acknowledgement
Note that the calculating code of this repository is heavily based on [Geo-PIFu](https://github.com/simpleig/Geo-PIFu). We thank the authors for their great job!

## Contact 
If you have some trouble using this software, please contact me[Jianchi Sun: [sunjc0306@mails.ccnu.edu.cn](mailto:sunjc0306@mails.ccnu.edu.cn) or [sunjc0306@qq.com](mailto:sunjc0306@qq.com).]



