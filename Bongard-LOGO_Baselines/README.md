# Bongard-LOGO-baselines

This repository contains the code to reproduce the evaluation results of different models
on our benchmark [Bongard-LOGO](https://arxiv.org/abs/2010.00763).

## System Requirements

* 64-bit Python 3.6 installation.
* PyTorch 1.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 16GB of DRAM. We recommend NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* Other dependencies: tqdm, pillow, yaml, tensorboardX, etc.

For more details, please check out the dockerfile in `docker` directory.

## Instructions
The `scripts` folder contains the exemplar scripts for starting different experiments.

##### Model Training:
```
cd scripts
bash run_[model_name]_model.sh
```
The [model_name] includes nine baselines:
* `cnn`: CNN-Baseline
* `wren`: [WReN-Bongard](https://arxiv.org/abs/1807.04225)
* `maml`: [ANIL](https://arxiv.org/abs/1909.09157)
* `meta`: [Meta-Baseline-SC](https://arxiv.org/abs/2003.04390)
* `meta_moco`: [Meta-Baseline-MoCo](https://arxiv.org/abs/2003.04390)
(Note: First put pre-trained encoder in the `materials` folder)
* `moco`: [MoCo](https://arxiv.org/abs/1911.05722)
* `protoNet`: [ProtoNet](https://arxiv.org/abs/1703.05175)
* `metaOptNet`: [MetaOptNet](https://arxiv.org/abs/1904.03758)
* `snail`: [SNAIL](https://arxiv.org/abs/1707.03141)

and also two training stages in the proposed [Meta-Baseline-PS](https://arxiv.org/abs/2010.00763):
* `program`: Pre-training the program synthesis module 
* `meta_prog`: Fine-tuning the meta-learner 
(Note: First make sure the pre-trained program-synthesis module is in the `materials` folder)


## Links to Datasets:
* *ShapeBongard_V2*: It contains 12,000 problems which are 
3,600 `free-form shape problems`, 4,000 `basic shape problems`, 
and 4,400 `abstract shape problems`.
We can download the dataset from here:
[[link]](https://drive.google.com/file/d/1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx/view?usp=sharing), and 
then unzip it into `materials` directory. 


* For ablation study, we also provide a variant of ShapeBongard_V2: 
*ShapeBongard_V2_FF*, which instead contains 12,000 `free-from shape problems`. 
We can download the dataset from here:
[[link]](https://drive.google.com/file/d/1Rf1yBOF_WbYJb0qTb5MabXWZ6hVDgIcS/view?usp=sharing), and then unzip it into `materials` directory.
(Note: Need to change the running scripts in `scripts` accordingly, in order to train with the new dataset)


## Reference

To cite this work, please use

```
@INPROCEEDINGS{Nie2020Bongard,
  author = {Nie, Weili and Yu, Zhiding and Mao, Lei and Patel, Ankit B and Zhu, Yuke and Anandkumar, Animashree},
  title = {Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020}
}

```


### Acknowledgement

This code is based on the repository [few-shot-meta-baseline](https://github.com/yinboc/few-shot-meta-baseline).
