# ASD Classification and Functional Connectivity Pattern Characterization Using rs-fMRI

A two-stage machine learning and explainable AI framework for studying Autism Spectrum Disorder (ASD) using resting-state functional Magnetic Resonance Imaging (rs-fMRI) data.

## Overview

This project explores ASD-associated brain connectivity patterns using data from the **Autism Brain Imaging Data Exchange (ABIDE-I)**. It combines machine learning, explainable artificial intelligence (XAI), and reference-based functional connectivity analysis to investigate both group-level differences and individual variation.

The framework consists of two complementary stages:

### Stage 1: ASD Classification and Explainable AI

- Extraction of functional connectivity features using the AAL-116 brain atlas.
- Mutual Information-based feature selection to identify informative connectivity features.
- ASD versus typically developing (TD) classification using Logistic Regression, RBF-SVM and other comparative models.
- Performance evaluation using standard classification metrics.
- SHAP-based explainability to identify influential brain connections and understand individual-level model predictions.

### Stage 2: Functional Connectivity Pattern Characterization

Extends the analysis beyond binary classification through two reference-based indices:

- **MRI ASD Pattern Index (I):** Quantifies the expression of a learned ASD-associated functional connectivity pattern relative to a TD reference distribution.
- **ASD Relative Expression Index (R):** Represents an individual's relative position within the ASD reference distribution.

The second stage also incorporates multisite consistency analysis, reliability assessment, covariate associations, alternative pattern comparisons, normative deviation analysis, exploratory subgroup analysis, and edge-level interpretation.

## Dataset and Feature Extraction

- **Dataset:** ABIDE-I
- **Imaging modality:** Resting-state fMRI
- **Preprocessing:** CPAC-preprocessed functional images (`filt_noglobal`)
- **Brain atlas:** Automated Anatomical Labeling (AAL-116)
- **Brain regions:** 116
- **Functional connectivity features:** 6,670 unique ROI-to-ROI connections

Functional connectivity is estimated using Pearson correlation between regional time series, followed by Fisher's z-transformation to construct participant-level connectivity feature vectors.

**Dataset:** [ABIDE-I Repository](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)

## Project Workflow

```text
             ABIDE-I Dataset
                    |
                    v
          Preprocessed rs-fMRI
                    |
                    v
           AAL-116 Parcellation
                    |
                    v
       Functional Connectivity Extraction
                    |
                    v
            6,670 FC Features
                    |
             +------+------+
             |             |
             v             v
          Stage 1        Stage 2
       Classification   Pattern Analysis
             |             |
       Feature Selection  Reference-Based
             |             Indices
       ML Classifiers       |
             |         Robustness and
        SHAP Analysis   Reliability Analysis
             |             |
             v             v
         ASD vs TD     Individual-Level
        Classification  Characterization
```

## Technologies Used

- Python
- NumPy and Pandas
- Scikit-learn
- SciPy
- Nilearn and NiBabel
- SHAP
- Matplotlib and Seaborn
- Statsmodels

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install numpy pandas scipy scikit-learn nilearn nibabel shap matplotlib seaborn statsmodels
```


## Key Areas

- Functional connectivity analysis
- Machine learning-based ASD classification
- Mutual Information-based feature selection
- Explainable artificial intelligence using SHAP
- Reference-based connectivity pattern indices
- Multisite neuroimaging analysis
- Individual-level pattern characterization
- Statistical and exploratory analysis
