# Classifying by Deforming: Classification As A Cross Domain Task Via Deformation 

This repository is the official implementation of [Classifying by Deforming: Classification As A Cross Domain Task Via Deformation ](https://arxiv.org/abs/2030.12345). 

## Requirements
1. Set up the environment
We recommend using a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Generation
To generate the synthetic dataset used in our experiments, run the following command:
```bash
cd data_gen
./generate_data.sh <number_of_batches> <output_directory>
```
Arguments:
- <number_of_batches>: The number of data batches to generate.
- <output_directory>: The target path where the dataset will be saved.

To generate our templates data, run the following command:
```bash
python data_gen/data_gen_src/template.py --template_dir /path/to/folder/saving_template --size <Image_resolution>
```

## Training
### Deformation Network
To train the model(s) in the paper, run this command:

1. Configuration
    
    Before training, please modify the configuration file at configs/train_config.yaml.
- Important: Update the data_path to point to your generated dataset directory.
- You can also adjust other hyperparameters (e.g., learning rate, batch size) in this file.

2. Run Training

**Option A: Train Cage Deformation Network**
```bash
cd cage_deformation
python src/train.py
```

**Option B: Train Grid Deformation Network**

```bash
cd grid_deformation
python src/train.py
```

### Deformation Metric Module

## Visualization
We provide a script to visualize the deformation effects of the **Cage-based model**. You can generate qualitative results on our synthetic dataset, MNIST, or Omniglot.

### Basic Usage

To run the visualization, use the following command:

```bash
python src/visualize.py --checkpoint-path <path_to_model> --output-dir <save_path> [options]
```

### Argument,Description:
```
--checkpoint-path: (Required) Path to the trained model weights. ⚠️ Note: This must be a cage-based model checkpoint.
--output-dir: Path to save the visualization results (default: images_eval).
--N: The number of samples to visualize (default: 8).
--show-cage: If set, the deformed cage grid will be visualized overlaid on the images.
--no-residual: If set, the model output will skip the residual flow, showing only the result of the Affine + Cage deformation.
--config-path: Path to the model configuration file (default: configs/train_config.yaml).
```

### Dataset Selection
You can choose which dataset to visualize using the following flags. If no dataset flag is provided, the script uses our Synthetic Dataset by default.
- (Default): Visualizes deformation on the Synthetic Dataset.

- `--mnist`: Visualizes deformation on the MNIST dataset.

- `--omniglot`: Visualizes deformation on the Omniglot dataset.

### Command Examples
1. Visualize Synthetic Data with Cage Overlay:
```bash
python  cage_deformation/src/test/model_vis.py \
  --checkpoint-path checkpoints/cage_model.pth \
  --output-dir results/synthetic_vis \
  --show-cage
```

2. Visualize MNIST (Affine + Cage only):
```bash
python  cage_deformation/src/test/model_vis.py \
  --checkpoint-path checkpoints/cage_model.pth \
  --output-dir results/mnist_vis \
  --mnist \
  --no-residual
```

3. Visualize Omniglot (16 samples):
```bash
python src/visualize.py \
  --checkpoint-path checkpoints/cage_model.pth \
  --output-dir results/omniglot_vis \
  --omniglot \
  --N 16
```

### Visualization Examples
Here, we present some visualization of our results.

<div align="center">
  <div style="display: flex; justify-content: space-between; width: 100%;">
    <img src="assets/vis_example1.png" width="32%">
    <img src="assets/vis_example2.png" width="32%">
    <img src="assets/vis_example3.png" width="32%">
  </div>
  <p align="center">Figure: Visualization Examples</p>
</div>


## Evaluation
`pending`
To evaluate my model on Omniglot, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models
`pending`
You can download pretrained models here:

- [Grid-based Model](https://drive.google.com/mymodel.pth) trained on our synthetic data using the configuration in `grid_deformation/configs/train_config.yaml`
- [Cage-based Model](https://drive.google.com/mymodel.pth) trained on our synthetic data using the configuration in `cage_deformation/configs/train_config.yaml`
- [Deformation Metric Module](https://drive.google.com/mymodel.pth) trained on our synthetic data use the ...


## Results
### Image Classification on Omniglot (Few shot)
We evaluate the generalization capability of our model on the **Omniglot** dataset under standard Few-Shot Classification settings.

The table below compares our method (pre-trained on **synthetic data**) against relevant baselines.

| Method | Pre-training Data | 5-way 1-shot Acc. | 5-way 5-shot Acc. |
| :--- | :--- | :---: | :---: |
| **Ours (SSL Pre-training)** | **Synthetic (Ours)** | **95.8%** | **97.9%** |

> 📋 **Note on Benchmarking:** \
> Since this project explores a novel setting using **Synthetic Data for Self-Supervised Learning**, there is no direct public leaderboard available. \
> We compare our method against **models trained from scratch** and **models pre-trained on real data** to demonstrate that our synthetic data pipeline achieves cross domain few-shot classification performance without requiring real-world annotations.

## Contributing
### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{Deform2025Class,
  title={Classifying by Deforming: Classification As A Cross Domain Task Via Deformation},
  author={Anonymous authors},
  year={2025}
}
```

### License
This project is under the MIT license. See [LICENSE](LICENSE) for details.