# ECG Image-Based Cardiac Condition Classification Using ML

A React + FastAPI research application with the supplied CardioLens UI, ECG-region
bounding boxes, automatic quarter-turn correction, twelve lead crops, validation,
and PCA/SVM classification. The existing trained model files are included.

## Vercel deployment

See [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md) to deploy the frontend and backend in one Vercel project. The upload limit is 4 MB.

## Open in VS Code

1. Extract the ZIP.
2. Open the `ECG-Project-Updated` folder in VS Code (the folder containing this README).
3. Use **Terminal → New Terminal**. Commands below are for Windows PowerShell.

Prerequisites: Python 3.12 and Node.js 22.12+ (Node 24 also works). Install Git if
publishing to GitHub. Existing project servers must be stopped with Ctrl+C before
starting another copy on ports 8000 and 5173.

## First-time setup

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm.cmd ci
cd ..
```

Explicit Python paths and `npm.cmd` work without changing the PowerShell execution
policy or activating the virtual environment.

## Run the backend — terminal 1

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health

## Run the frontend — terminal 2

Open a second terminal from the project root:

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open http://127.0.0.1:5173/ in your browser. Keep both terminals running. Use the
127.0.0.1 address consistently. Press Ctrl+C in each terminal to stop its server.

After first-time setup, you can also run `run_backend.bat` and `run_frontend.bat`.
In VS Code, **Terminal → Run Task** offers **ECG: Backend** and **ECG: Frontend**.
Run both tasks.

## Use the application

Upload JPG/JPEG/PNG (up to 4 MB, minimum 300 pixels on each side). The server checks
file bytes and locates the ECG panel before extraction. Some downloaded JPG files
actually contain WebP; the decoder handles this and displays a warning. Inspect
the green boundary, upright ECG and twelve lead crops. Use the 180-degree correction
if needed. Review the lead orientation/labels and then run the classifier.

Lead labels are assigned by standard layout order, not OCR. Complex layouts,
overlapping traces, heavy perspective distortion or unclear panels may fail.
Validation and model-confidence gates remain active. Outputs are for research,
not diagnosis. The About page contains evaluation figures supplied with the UI;
they are not recalculated from each upload.

## Verification

```powershell
cd frontend
npm.cmd run build
npm.cmd run lint
node --test src/validation.test.js
cd ..
```

Optional integration checks, with the backend running and the two original sample
files in a folder you select:

```powershell
.\.venv\Scripts\python.exe validation/test_upload_pipeline.py --samples-dir "C:\path\to\sample-images"
```

This optional check expects `risk fellow.jpg` and `unknown.jpg`. These sample
images are not included in this package.

## Project folders

- `frontend/`: CardioLens UI, themes, pages, upload validation and frontend tests.
- `backend/main.py`: health, validation and prediction API.
- `preprocessing/`: ECG-region localization, orientation, layout and extraction checks.
- `models/`: PCA, classifier and calibrated validation thresholds.
- `validation/`: optional live API integration checks.
- `.vscode/`: backend/frontend run tasks.
- `CHANGES.md`: files and behavior added or updated.

Dependency versions match the working environment. The classifier was originally
saved with scikit-learn 1.9.0; the working environment uses 1.9.1, which can emit a
model-version warning. Clinical validity is not established by these software tests.

The package excludes virtual environments, node_modules, build output, scanner
sample images, debug images, logs, backups and the original training dataset.
Retraining is not required to run the included model.

## Publish a new repository to HarshithaSowmya

The suggested repository name is:
`ECG-Image-Based-Cardiac-Condition-Classification-Using-ML`

The ZIP itself is named with your requested full title. To publish from VS Code:

1. Open the extracted project folder.
2. Open Source Control and choose **Initialize Repository**.
3. Review the files, stage them, and commit with a message such as
   `Add ECG classification UI and validation pipeline`.
4. Choose **Publish to GitHub** and sign in as **HarshithaSowmya**.
5. Enter the repository name above and choose the visibility you want.

If you have already created an empty GitHub repository with that name, you can use
these commands instead, from the extracted project root:

```powershell
git init
git add .
git commit -m "Add ECG classification UI and validation pipeline"
git branch -M main
git remote add origin https://github.com/HarshithaSowmya/ECG-Image-Based-Cardiac-Condition-Classification-Using-ML.git
git push -u origin main
```

Git will request authentication if necessary. The .gitignore excludes local
configuration, installed dependencies and uploads. Do not upload patient images.
