
# KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance

This repository contains the implementation of the approach proposed in the paper "KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance".

## Overview

KAIROS is a system designed to enhance intrusion detection and facilitate comprehensive investigations by leveraging whole-system provenance data. By analyzing the causal relationships between system events, KAIROS aims to provide a practical solution for identifying and understanding security breaches.

## Features

- **Intrusion Detection**: Utilizes provenance data to detect anomalous activities indicative of security threats.
- **Investigation Tools**: Offers tools to trace and analyze the sequence of events leading to and following an intrusion.
- **Comprehensive Analysis**: Provides a holistic view of system operations to aid in understanding complex attack vectors.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages listed in `DARPA/settings/`

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nahidul-opu/COMP7860-Kairos.git
   cd COMP7860-Kairos
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
You may need to manually install PyTorch based on your workstation configuraion.

# Author's Note:


This repository contains the implementation of the approach proposed 
in the paper 
"_KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance_".

Please cite this paper if you use the model or any code
from this repository in your own work:
```
@inproceedings{cheng2024kairos,
  title={KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance},
  author={Cheng, Zijun and Lv, Qiujian and Liang, Jinyuan and Wang, Yang and Sun, Degang and Pasquier, Thomas and Han, Xueyuan},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  year={2024},
  organization={IEEE}
}
```

We provide a [demo](DARPA/README.md)
to illustrate step-by-step
how you can run the code end-to-end.
Additionally, we provide IPython notebook
scripts for all of our experiments.
> Due to the extended amount of time it takes to
> train a model, we also provide pre-trained models
> of our experimental datasets.
> You can download these models directly from our [Google Drive](https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C).

Our paper and [the supplementary material](supplementary-material.pdf)
contain links to all publicly available datasets used in our experiments.
