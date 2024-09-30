# SmartGD: A GAN-Based Graph Drawing Framework for Diverse Aesthetic Goals
This repo contains a simple demonstration for the IEEE TVCG‘2024 paper entitled "[SmartGD: A GAN-Based Graph Drawing Framework for Diverse Aesthetic Goals](https://ieeexplore.ieee.org/document/10224347)". It includes:

* a dataloader for Rome Graphs dataset,
* a basic implementation of SmartGD model,
* a generator checkpoint trained for Stress majorization,
* and a demo notebook for model traning and evaluation.

## Environment
This code has been tested on `python3.11` + `cuda12.1` + `pytorch2.4` + `pyg2.4`. 

## Configuration
The default hyper-parameters of the model have been configured to reproduce the best performance reported in the [SmartGD paper](https://ieeexplore.ieee.org/document/10224347). 

## Training & Evaluation
* This repo provides a demo notebook `smartgd_demo.ipynb` for model training and evaluation. With Nvidia A100, each training epoch takes 5 minutes on average. It takes up to 1000 epochs to completly converge.

* This repo includes a model checkpoint `generator_493.pt`, which reproduces the result for Stress Majorization-only objective function reported in the paper.

* For evaluation on custom data, the easiest way is to subclass `RomeDataset` and override `raw_file_names` and `process_raw` methods.
    > **Caveat**: Even though the behavior of `process` do not need to be overriden, it is required to have a dummy `def process(self): super().process()` defined in the subclasses to make it work properly. For details, please refer to `pyg.data.InMemoryDataset` [documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset).

## Citation
If you used our code or find our work useful in your research, please consider citing:
```
@ARTICLE{10224347,
  author={Wang, Xiaoqi and Yen, Kevin and Hu, Yifan and Shen, Han-Wei},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={SmartGD: A GAN-Based Graph Drawing Framework for Diverse Aesthetic Goals}, 
  year={2024},
  volume={30},
  number={8},
  pages={5666-5678},
  keywords={Layout;Graph drawing;Deep learning;Generative adversarial networks;Stress;Generators;Training data;Deep learning for visualization;generative adversarial networks;graph visualization},
  doi={10.1109/TVCG.2023.3306356}}
```
