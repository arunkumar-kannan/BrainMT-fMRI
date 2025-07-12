<div align="center">

# BrainMT: A Hybrid Mamba‑Transformer Architecture for Modeling Long‑Range Dependencies in Functional MRI Data

**[Arunkumar Kannan](https://arunkumar-kannan.github.io/), [Martin A. Lindquist](https://sites.google.com/view/martinlindquist/home), [Brian Caffo](https://sites.google.com/view/bcaffo/home)** 

Johns Hopkins University

[![arXiv](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.22591)

</div>

BrainMT has been accepted to [MICCAI'25](https://conferences.miccai.org/2025/en/)


## ✨ Highlights

🔍 **Motivation:** Can we develop deep learning models that efficiently operate on voxel-level fMRI data - just like we do with other medical imaging modalities?

🧠 **Architecture:** We introduce BrainMT, a novel hybrid framework designed to efficiently learn and integrate long-range spatiotemporal attributes in fMRI data. BrainMT framework operates in two stages:
  - 1️⃣ A bidirectional Mamba block with a temporal-first scanning mechanism to capture global temporal interactions in a computationally efficient manner; and
  - 2️⃣ A transformer block leveraging self-attention to model global spatial relationships across the deep features processed by the Mamba block.
    
📈 **Results:** Through extensive experiments and ablation studies on two large-scale public datasets - UKBioBank (UKB) and the Human Connectome Project (HCP), we demonstrate that BrainMT outperforms existing methods and generalizes robustly across diverse tasks for improved phenotypic prediction in neuroimaging.

<div align="center">
  <img src="assets/teaser.png" width="70%" alt="BrainMT Teaser Figure"/>
</div>

### 📜 Abstract

*Recent advances in deep learning have made it possible to predict phenotypic measures directly from functional magnetic resonance imaging (fMRI) brain volumes, sparking significant interest in the neuroimaging community. However, existing approaches, primarily based on convolutional neural networks or transformer architectures, often struggle to model the complex relationships inherent in fMRI data, limited by their inability to capture long-range spatial and temporal dependencies. To overcome these shortcomings, we introduce BrainMT, a novel hybrid framework designed to efficiently learn and integrate long-range spatiotemporal attributes in fMRI data. Our framework operates in two stages: (1) a bidirectional Mamba block with a temporal-first scanning mechanism to capture global temporal interactions in a computationally efficient manner; and (2) a transformer block leveraging self-attention to model global spatial relationships across the deep features processed by the Mamba block. Extensive experiments on two large-scale public datasets, UKBioBank and the Human Connectome Project, demonstrate that BrainMT achieves state-of-the-art performance on both classification (sex prediction) and regression (cognitive intelligence prediction) tasks, outperforming existing methods by a significant margin.*


## ✅ To‑Do List for Code Release

- [x] ~~Create repository~~     
- [ ] **Installation guide** – provide `requirements.txt` / `environment.yml` and setup instructions  
- [ ] **Training scripts** – release reproducible training pipeline (`train.py`, configs, SLURM examples)  
- [ ] **Evaluation scripts** – include scripts for validation and test‑set evaluation  
- [ ] **Benchmark tables** – add performance tables & plots in `docs/`  
- [ ] **Inference demo** – provide an interactive notebook for single‑subject inference  
- [ ] **Dataset prep** – share preprocessing scripts
- [ ] **Config files** – upload YAML config templates for different tasks


