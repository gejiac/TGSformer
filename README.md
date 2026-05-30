# TGSformer: A TSK-Guided Graph-Spectral Transformer for Interpretable EEG-based Alzheimer's Disease Classification

This repository contains the official implementation of **TGSformer**, a **Takagi--Sugeno--Kang (TSK)-guided Graph-Spectral Transformer** for interpretable electroencephalogram (EEG)-based Alzheimer's disease (AD) classification.

TGSformer embeds TSK fuzzy rules into the encoder layers, where rule activations directly participate in intermediate representation learning. The model jointly captures temporal dependencies, latent graph relations, and frequency-domain dynamics from EEG signals.

![Overall architecture of TGSformer](TGSformer/overall.png)

## Highlights

* **TSK-guided representation learning**
  TSK rules are embedded inside the encoder instead of being used only as a final fuzzy classifier.

* **TSK-Guided Adaptive Graph Convolution (TAGC)**
  TAGC integrates sample-specific dynamic graphs with learnable rule graph templates to model latent feature relations.

* **TSK-Guided Adaptive Spectral Filtering (TASF)**
  TASF dynamically aggregates rule-conditioned complex spectral filters according to the spectral context of each EEG sample.

* **Rule-constrained graph-spectral collaborative learning**
  Rule confidence and rule balance constraints improve the clarity, diversity, and stability of TSK rule learning.

* **Interpretable EEG classification**
  The model supports gradient-based saliency visualization and class-wise TSK rule activation analysis.

## Method Overview

Given an EEG segment, TGSformer first maps the input into token representations. These tokens are then processed by stacked encoder layers. Each encoder layer contains two complementary TSK-guided branches:

1. **TAGC** models adaptive latent graph relations.
2. **TASF** performs sample-specific spectral filtering.

The two branches are recurrently refined and fused inside the encoder. During training, auxiliary TSK rule constraints are introduced to encourage confident sample-level rule activation and balanced batch-level rule usage.

## Datasets

The experiments are conducted on three EEG-based AD classification settings:

| Dataset  | Task                             | Classes       |
| -------- | -------------------------------- | ------------- |
| ADFTD    | Three-class classification       | AD / FTD / NC |
| ADFTD-BI | Binary classification from ADFTD | AD / NC       |
| APAVA    | Binary classification            | AD / NC       |

All experiments follow a **cross-subject** evaluation protocol. EEG segments from the same subject appear only in one of the training, validation, or test sets.

## Requirements

```bash
python >= 3.8
torch >= 1.13
numpy
scipy
scikit-learn
pandas
matplotlib
mne
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please organize the dataset as follows:

```text
dataset/
├── ADFTD/
│   ├── Feature/
│   └── Label/
└── APAVA/
    ├── Feature/
    └── Label/
```

Each EEG sample is segmented into fixed-length windows. In our experiments, the input length is set to 256 time points.

## Training

Example command for ADFTD:

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --model TGSformer \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
```

Example command for APAVA:

```bash
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --model TGSformer \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des Exp \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
```

## Evaluation

The model is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* AUROC
* AUPRC

For the three-class ADFTD task, macro-average metrics are reported. All final results are reported as the mean and standard deviation over five random seeds.

## Interpretability

TGSformer provides two types of interpretability analysis.

### Gradient-based EEG Topographic Visualization

Channel-level saliency is computed by taking the gradient of the class score with respect to the input EEG signal. The saliency scores are then projected onto the standard EEG montage to visualize class-related spatial patterns.

### TSK Rule Activation Visualization

The firing strengths of TSK rules are recorded from both the graph branch and the spectral branch. Class-wise rule activation distributions are used to analyze how different classes activate different latent rule patterns.

## Main Results

TGSformer achieves competitive performance across three EEG-based AD classification settings. It shows clear advantages on the ADFTD three-class task and the APAVA binary task, while remaining competitive on the ADFTD-BI task.

The results demonstrate that combining TSK-guided adaptive graph relation modeling with adaptive spectral filtering can improve both classification performance and rule-level interpretability.

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{tgsformer2026,
  title={A TSK-Guided Graph-Spectral Transformer for Interpretable EEG-based Alzheimer's Disease Classification},
  author={Your Name and Co-authors},
  journal={Under Review},
  year={2026}
}
```

## License

This project is released for academic research purposes. Please check the license file for more details.

## Acknowledgement

This implementation is developed for EEG-based Alzheimer's disease classification research. Parts of the experimental protocol follow the medical time-series classification setting used in Medformer.
