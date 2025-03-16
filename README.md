# MVA_GenAI_FP_Learning_Gradients
MVA class 2024-2025. Final Project of GenAI class : "learning gradients of convex functions with monotone gradient networks"

## Description
Final project based on article : https://arxiv.org/abs/2301.10862

The task is to learn gradients of convex functions with monotone gradient networks. Two neural networks architectures are proposed and studied.

## Installation
### Prerequisites
Ensure you have Python 3.x installed. You can check by running:
```bash
python --version
```

### Create a Virtual Environment
#### Using venv
```bash
python -m venv venv
source venv/bin/activate
```

#### Using Conda
```bash
conda env create -f environment.yml
conda activate learning_gradients
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
```bash
python start.py
```

## Running Tests
Test algorithms:
```bash
python -m test.tests_algos
```

Test Kernels
```bash
python -m test.tests_kernels
```