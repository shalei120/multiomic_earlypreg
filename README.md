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
The repository does not include example datasets. Provide your own files matching
the expected formats:

* ``cfRNA`` file – two columns with either nucleotide sequences and binary labels
  or gene identifiers with expression counts.
* ``cfDNA`` file – two columns with nucleotide sequences and binary labels.

Train the transformer model with:

```bash
python src/training/train_transformer.py --cfrna <cfRNA.csv> \
       --cfdna <cfDNA.csv> --epochs 5 --plot roc.png
```

The command reports the AUROC and saves ``roc.png`` if a plot path is provided.

## Early pregnancy tabular data

This repository also provides a simplified tabular example. You must supply three CSV files:

* ``cfDNA`` labels – sample identifiers with binary labels
* ``cfRNA`` labels – sample identifiers with binary labels
* ``FJ_02.All_sample_reads_count.csv`` containing gene counts per sample

Run the MLP classifier with:

```bash
python src/training/train_tabular.py \
  --features <features.csv> \
  --cfrna <cfRNA_labels.csv> \
  --cfdna <cfDNA_labels.csv> --epochs 20 --plot roc_tabular.png
```

The script reports the AUROC after training and saves a ROC curve if a plot path is provided.
