# Multi-Omic Early Pregnancy

This repository demonstrates a prototype implementation of a sequence-based transformer model for integrating cfRNA and cfDNA data in preterm birth (PTB) risk prediction.

## Requirements

- Python 3.11
- PyTorch

Install dependencies with:

```bash
pip install torch numpy scikit-learn matplotlib pandas
```

## Usage

Run the training script with paths to your preprocessed cfRNA and cfDNA data.
Example cfRNA and cfDNA files are included in the ``data`` directory.
The ``sample_cfrna.csv`` file was generated from an Excel sheet and contains two
columns (gene identifier and count):

```text
data/sample_cfrna.csv
data/sample_cfdna.csv
```

Launch a training run on the sample data with:

```bash
python src/training/train_transformer.py --cfrna data/sample_cfrna.csv \
       --cfdna data/sample_cfdna.csv --epochs 5 --plot roc.png
```

This command trains the model and saves ``roc.png`` containing the classification ROC curve.

Replace ``data/sample_cfrna.csv`` with a file containing gene identifiers and expression counts. ``data/sample_cfdna.csv`` should contain nucleotide sequences and binary labels.

## Early pregnancy tabular data

This repository also includes a simplified tabular example. The files
`cfDNA_early.csv` and `cfRNA_early.csv` store sample identifiers with
corresponding labels, while `FJ_02.All_sample_reads_count.csv` provides gene
counts for each sample.

Train the MLP classifier on this data with:

```bash
python src/training/train_tabular.py \
  --features data/FJ_02.All_sample_reads_count.csv \
  --cfrna data/cfRNA_early.csv \
  --cfdna data/cfDNA_early.csv --epochs 20 --plot roc_tabular.png
```

The script reports the AUROC after training and saves a ROC curve if a path is
specified.

The CSV files in `data` are intentionally empty placeholders. Replace them with your own preprocessed datasets before running the training scripts.
