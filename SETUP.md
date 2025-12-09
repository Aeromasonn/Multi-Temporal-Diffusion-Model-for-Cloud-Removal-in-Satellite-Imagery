This document explains how to set up the environment, prepare data, and run the core components of this project.

1. Clone the repository
```
git clone https://github.com/Aeromasonn/Compsci372-Final_Project.git
cd Compsci372-Final_Project
```

2. Create Conda Environment

This project does not rely on unusual or difficult-to-install libraries, so replicating the entire environment is not strictly necessary.
However, for consistency, you can create the environment from the provided file:

```
conda env create -f environment.yml
conda activate base
```

3. Data Preparation
a) Download the raw Sen2_MTC dataset

Download from:
https://drive.google.com/file/d/1-hDX9ezWZI2OtiaGbE8RrKJkN1X-ZO1P/view

Place it under:
```
<YOUR_PROJECT_DIRECTORY>/Sen2_MTC
```

Sample directory structure:
```
<YOUR_PROJECT_DIRECTORY>
└── Sen2_MTC
    └── dataset
        └── Sen2_MTC
            ├── T09WWP_R071
            ├── T12TUR_R027
            ├── ...
```
b) (Optional but recommended) Download the precomputed Sen2_MTC dataset (.pt)

Download from:
https://drive.google.com/file/d/1CmIEcsDwbfReT8ApG0sIZ2Hw5mECJKyn/view?usp=sharing

This dataset contains precomputed and packed train/val/test loaders.

Place it under:
```
<YOUR_PROJECT_DIRECTORY>/Ckpts


Sample directory structure:

<YOUR_PROJECT_DIRECTORY>
└── Ckpts
    └── Sen2MTC_FULL_3v1_norm.pt
```

4. Essential project layout:
```
<YOUR_PROJECT_DIRECTORY>
├── docs
│   └── reference papers
│
├── Downstream
│   └── Downstream_Task_Final.ipynb         # A potential downstream application
│
├── Model
│   ├── Dataloader.py                       # All dataloader programs
│   ├── evaluation.py                       # Evaluation methods
│   ├── get_pretrained.py                   # Initialize pretrained models
│   ├── model.py                            # Core model architecture
│   ├── trainers.py                         # Trainer loop
│   └── utils.py                            # Utility functions
│
├── Notebooks
│   ├── Evaluation.ipynb                    # Evaluate the model using pretrained results
│   ├── Pipeline_final_local.ipynb          # Local training pipeline
│   └── Pipeline_final_via_pretrained.ipynb # Pre-packed dataset training pipeline (Colab-friendly)
│
├── pretrained
│   └── *.pt                                # All pretrained models
│
├── Test_scripts
│   └── ...                                 # Testing scripts for exploration
│
├── README.md
└── environment.yml
```