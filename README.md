
# KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance

This repository contains the implementation of the approach proposed in the paper "KAIROS: Practical Intrusion Detection and Investigation using Whole-system Provenance".

## Overview

KAIROS is a system designed to enhance intrusion detection and facilitate comprehensive investigations by leveraging whole-system provenance data. By analyzing the causal relationships between system events, KAIROS aims to provide a practical solution for identifying and understanding security breaches.

## Features

- **Intrusion Detection**: Utilizes provenance data to detect anomalous activities indicative of security threats.
- **Investigation Tools**: Offers tools to trace and analyze the sequence of events leading to and following an intrusion.
- **Comprehensive Analysis**: Provides a holistic view of system operations to aid in understanding complex attack vectors.

## Module Overview
Kairos builds a separate model for each dataset, with each dataset requiring specific preprocessing, training, and testing phases. This project contains of the following top level modules:
- **Unix Environment**
  - [DARPA TC CADETS Engagement 3](DARPA/CADETS_E3)
  - [DARPA TC CADETS  Engagement 5](DARPA/CADETS_E5)
  - [DARPA TC CLEARSCOPE  Engagement 3](DARPA/CLEARSCOPE_E3)
  - [DARPA TC CLEARSCOPE  Engagement 5](DARPA/CLEARSCOPE_E5)
  - [DARPA TC OpTC](DARPA/OpTC)
  - [DARPA TC THEIA  Engagement 3](DARPA/THEIA_E3)
  - [DARPA TC THEIA  Engagement 5](DARPA/THEIA_E5)
  - [Stream Spot](StreamSpot)
- **Windows Environment**
  - [CADETS  Engagement 3](DARPA/WINDOWS/CADETS_E3)
  - [CADETS  Engagement 5](DARPA/WINDOWS/CADETS_E5)
  - [CLEARSCOPE  Engagement 5](DARPA/WINDOWS/CLEARSCOPE_E5)
## Getting Started


### Prerequisites

- Anaconda
- Python 3.9
- PostgreSQL
- GraphViz
- CUDA

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nahidul-opu/COMP7860-Kairos.git
   cd COMP7860-Kairos
   ```

2. **Virtual Environment Set Up**:

   Follow the Python runtime environment setup instructions written in [Anaconda Python Environment](DARPA/settings/environment-settings.md)
3. **Database Set Up**:

   The preprocessing steps involve storing the parsed logs into a database for each dataset. To work with a specific dataset, you need to create a schema specific for that dataset. Instructions for creating the database is located in [PostgreSQL Database](DARPA/settings/database.md)

# Dataset Download
- [Download DARPA TC CADETS Engagement 3](https://drive.google.com/drive/u/0/folders/179uDuz62Aw61Ehft6MoJCpPeBEz16VFy)
- [Download DARPA TC CADETS  Engagement 5](https://drive.google.com/drive/u/0/folders/1YOaC0SMGjBnrT9952EwmKKngQkBYf4hY)
- [Download DARPA TC CLEARSCOPE  Engagement 3](https://drive.google.com/drive/u/0/folders/1cbOHa5_dlu0XF8od5YKKqCGOawHzqaT_)
- [Download DARPA TC CLEARSCOPE  Engagement 5](https://drive.google.com/drive/u/0/folders/1S-LrRdu1tCjUMQA_VdKj_OXWs4BA7Hk_)
- [Download DARPA TC OpTC](https://drive.google.com/drive/u/0/folders/1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa)
- [Download DARPA TC THEIA  Engagement 3](https://drive.google.com/drive/u/0/folders/1AWXy7GFGJWeJPGzvkT935kTfwBYzjhfC)
- [Download DARPA TC THEIA  Engagement 5](https://drive.google.com/drive/u/0/folders/13zdJvC62zsJc2nD7KWxtN9xkk05LdQGw)
- [Download Stream Spot](https://github.com/sbustreamspot/sbustreamspot-data/blob/master/all.tar.gz)

# Models
Pretrained models shared by the original authors can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C)