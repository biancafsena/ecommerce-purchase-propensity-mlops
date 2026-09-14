# Raw data

This directory contains immutable source data used by the purchase propensity pipeline.

## Online Shoppers Purchasing Intention

- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 468
- **DOI:** https://doi.org/10.24432/C5F88Q
- **License:** Creative Commons Attribution 4.0 International
- **Original file:** `online_shoppers_intention.csv`
- **Instances:** 12,330 browsing sessions
- **Predictive features:** 17
- **Target:** `Revenue`
- **SHA-256:** `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`

The target indicates whether a browsing session ended in a purchase. According to the
dataset documentation, 10,422 sessions belong to the negative class and 1,908 sessions
belong to the positive class.

## Data governance

Raw data must not be manually edited. The source file is tracked by DVC and intentionally
excluded from Git.

To acquire and verify the dataset from its official source, run:

```bash
poetry run python -m purchase_propensity.data.download
