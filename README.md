# Customer Segmentation: ATM-to-ECOM Debit Adoption

This project predicts whether an eligible customer will make a successful ECOM transaction using a debit card in the 30 days after a monthly snapshot.

## Cohort and label

- Eligible: at least one successful ATM transaction using a debit card in the preceding 90 days.
- Excluded from that snapshot's cohort: any successful ECOM transaction using a debit card in the preceding 90 days.
- Target: at least one successful ECOM debit-card transaction in the following 30 days.
- The model estimates propensity. A randomized campaign control group is still required to measure incremental campaign impact.

## Project folders

- `SQL/`: schema inspection and customer-level feature/label query.
- `Python/`: database connection, CSV export, model tuning, and final holdout evaluation scripts.
- `Config/`: local database configuration and a safe example template.
- `Data/`: local customer-level CSV exports (ignored by Git) and aggregate model reports.
- `Models/`: saved tuning candidate and metadata.
- `Architecture/`: project architecture image.
- `Logs/`: local application logs.

## Setup

PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
Copy-Item .\Config\config.example.yaml .\Config\config.yaml
```

Fill in `Config/config.yaml` with the MySQL connection values. Do not commit that file or share its credentials.

## Data and model workflow

The existing feature dataset is `Data/ecom_debit_training_dataset.csv`. It is intentionally excluded from Git because it contains customer-level records. The original broad extract `Data/customer_training_dataset.csv` is also retained locally as a backup and excluded from Git.

To rebuild the ECOM-debit dataset from MySQL (a potentially long query):

```powershell
python .\Python\export_ecom_debit_dataset.py
```

To tune model parameters without evaluating the final October holdout:

```powershell
python .\Python\tune_ecom_debit_models.py
```

After choosing and freezing the model, evaluate the reserved October holdout once:

```powershell
python .\Python\evaluate_ecom_debit_test.py
```

Tuning results and final holdout metrics are aggregate reports and are tracked in Git. Customer-level CSVs and local credentials are not.
