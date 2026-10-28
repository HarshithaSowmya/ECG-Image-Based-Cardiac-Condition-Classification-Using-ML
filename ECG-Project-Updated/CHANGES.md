# Changes in this package

## Added

- `preprocessing/ecg_region.py`: locate the ECG panel, reject sparse form/table
  rules, correct quarter-turn orientation, generate a green boundary preview and
  twelve lead-region previews.
- `frontend/src/components/`: supplied Hero and reusable UI components.
- `frontend/src/pages/`: supplied About and Limits pages; integrated Analysis page.
- `frontend/src/data.js`: supplied themes and project-page data.
- `frontend/src/validation.js` and `validation.test.js`: upload/content and API
  response validation, with automated checks.
- `frontend/public/background.jpg` and `heart-background.jpg`: supplied artwork.
- `validation/test_upload_pipeline.py`: optional live-API sample checks.
- `.vscode/tasks.json`, run scripts, this README, and pinned runtime requirements.

## Updated

- `frontend/src/App.jsx`, `App.css`, `index.css`: CardioLens navigation, three
  themes, heartbeat-speed animation, project pages, solid surfaces and square
  corners. Upload state survives navigation between pages.
- Analysis workflow: real server checks, original/boundary/upright views,
  180-degree correction, twelve lead crops, explicit extraction review, class
  probabilities and CSV export.
- `backend/main.py`: rotation query validation and integration with region-aware
  preprocessing. Existing model-confidence gate remains active.
- `preprocessing/ecg_validator.py`: region isolation before lead extraction;
  monochrome-grid and broad-waveform handling; extraction metadata; safe decoding
  of WebP content downloaded under a JPG/PNG filename.
- `frontend/index.html`: CardioLens page title and metadata.

## Retained

The trained PCA/classifier artifacts, calibrated thresholds and underlying
contour-based feature representation were retained. The bundled UI's dataset and
performance figures are supplied project content, not a new model evaluation.

The software checks run during development covered the frontend build/lint,
client upload checks, complete and incomplete ECG layouts, ordinary non-ECG
images, line-chart rejection, both supplied mixed notes/ECG examples, PNG uploads,
manual rotation, API errors and prediction confidence gating.
