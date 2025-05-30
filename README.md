# Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms
Authors: [Hiroshi Kera](https://researchmap.jp/h_kera)\*, [Nico Pelleriti](https://www.zib.de/de/members/pelleriti)\*, [Yuki Ishihara](https://researchmap.jp/yishihara?lang=en), [Max Zimmer](https://maxzimmer.org/), [Sebastian Pokutta](https://www.pokutta.com/) (*equal contribution)


This repository provides code for reproducing the experiments of our paper "[Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms](https://arxiv.org/abs/2505.23696)". The code is based on PyTorch 2.5 and SageMath 10.0+. 

## SageMath Setup

SageMath 10.0+ cannot be currently installed using `apt-get` and `pip install` (May, 27, 2025). 
Follow the instruction in [this page](https://sagemanifolds.obspm.fr/install_ubuntu.html). 

## Reproducing main experiements
For dataset generation, 
```sh
bash sh/generate_getaset_sweep.sh
```

The number of tokens in infix and monomial representations can be measured and plotted by 
```sh
python scripts/analysis/infix_vs_monomial_token_count_and_plot.py
```
This gives Figure 7.

For training models, 
```sh
bash sh/train_sweep.sh
```

For Table 1,
```sh
python scripts/analysis/model_evaluation.py
```

For Table 2, you require the pretrained model from the previous step for the field $\mathcal{F}_{31}$ and $5$ leading terms. For $n=5$ variables, replace the save_path in `sweeps/n=5/evaluation.yaml` by the path of your saved model. You also require the generated datasets from the previous step.

Then run 

```sh
wandb sweep sweeps/n=5/evaluation.yaml
```

and execute the wandb agent. The setup for $n=3$ and $n=4$ is near identical. 

To obtain the values for Figure 3, replace again save_path by the path to the model and then generate the wandb sweep for `sweeps/n=4/ood_evaluation.yaml`. 

## Reproducing others
The comparison of infix and monomial representations in the cumulative product task (Table 3) can be reproduced by 
```sh
bash train_infix_vs_monomial.sh
```


## General Usage

For dataset generation, 
1. prepare a config file in `config/problems/`
2. prepare the script `sh/generate_getaset.sh` and run it.
```sh
bash sh/generate_getaset.sh
```

For training, 
1. prepare a config file in `config/experiments/`
2. prepare the script `sh/train.sh` and run it.
```sh
bash sh/train.sh
```
- `--dryrun` allows you to walk through the experiments with small number of samples. The wandb log is sent to project named dryrun. 

For OBBA, 
1. update the `save_path` in the relevant sweep config (e.g., `sweeps/n=5/evaluation.yaml`) to your trained model path.
2. run the sweep via wandb. 

## Project Structure

```
Transformer-BB/
├── config/         # Configuration files
├── data/           # Dataset and preprocessing scripts
├── figs/           # Figures and plots
├── notebook/       # Jupyter notebooks
├── results/        # Experimental results
├── scripts/        # Script files
├── sh/             # Shell scripts
├── src/            # Source code
├── sweeps/         # wandb yaml sweep files 
└── tests/          # Test cases and unit tests
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{kera_pelleriti2025computational,
  title={Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms},
  author={Hiroshi Kera and Nico Pelleriti and Yuki Ishihara and Max Zimmer and Sebastian Pokutta},
  year={2025},
  archivePrefix={arXiv},
  eprint={2505.23696}
}
```
