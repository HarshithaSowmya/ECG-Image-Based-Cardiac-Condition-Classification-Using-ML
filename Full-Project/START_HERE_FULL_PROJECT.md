# Full ECG project with original data and outputs

This package combines the original heart disease (2).zip with the latest updated
application. All original project data folders and training scripts are included.
Git LFS pointer files were replaced, where available, with original public output
contents verified against the SHA-256 hashes in your ZIP. See FULL_PACKAGE_REPORT.json
for exact counts and any unresolved references.

## Run locally from this folder

Install Python 3.12 and Node.js 22.12 or newer.

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
    cd frontend
    npm.cmd ci
    cd ..

Terminal 1:

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Terminal 2:

    cd frontend
    npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort

Open http://127.0.0.1:5173/.

## Training

Training code and its existing feature/metric CSVs are in training/. Run scripts
from this project root. Training changes model artifacts, so keep the included
trained models as a backup before retraining.

    .\.venv\Scripts\python.exe -m training.create_dataset
    .\.venv\Scripts\python.exe -m training.prepare_data
    .\.venv\Scripts\python.exe -m training.model_selection
    .\.venv\Scripts\python.exe -m training.final_model

Installed virtual environments/node_modules, caches, machine logs and .DS_Store
were excluded. They are recreated by the setup commands. Datasets, training code
and preprocessing outputs were not intentionally omitted.

The huge training/output folders are for local research; do not bundle them into
the deployed Vercel inference function. Use the smaller application directory in
the GitHub repository for deployment.
