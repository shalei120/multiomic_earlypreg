# Multi-Omic Early Pregnancy

This repository demonstrates a prototype implementation of a sequence-based transformer model for integrating cfRNA and cfDNA data in preterm birth (PTB) risk prediction.

## Requirements

- Python 3.11
- PyTorch

Install dependencies with:

```bash
pip install torch numpy scikit-learn matplotlib
```

## Usage

Run the training script with paths to your preprocessed cfRNA and cfDNA data.
Example CSV files are included in the ``data`` directory:

```text
data/sample_cfrna.csv
data/sample_cfdna.csv
```

Launch a training run on the sample data with:

```bash
python src/multiomics_transformer.py --cfrna data/sample_cfrna.csv \
       --cfdna data/sample_cfdna.csv --epochs 5 --plot roc.png
```

This command trains the model and saves ``roc.png`` containing the classification ROC curve.

Replace the sample CSVs with your own files formatted as ``sequence,label`` to use real data.
