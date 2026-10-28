# Deploy the full application on Vercel

The React frontend and FastAPI backend run in ONE Vercel project. The browser
calls the same-origin `/api/health`, `/api/validate` and `/api/predict` endpoints.
No Render service or backend URL is needed. Uploads are capped at 4 MiB to leave
multipart overhead below Vercel's 4.5 MB request-body ceiling.

## GitHub import settings

1. In Vercel choose Add New â†’ Project and import
   `HarshithaSowmya/ECG-Image-Based-Cardiac-Condition-Classification-Using-ML`.
2. Select branch `ecg-ui-validation`, or merge pull request #2 and use main.
3. Set **Root Directory** to **ECG-Project-Updated**. Do not select frontend or Deployment.
4. The included `vercel.json` defines:
   - Framework: Vite
   - Install: `npm --prefix frontend ci`
   - Build: `npm --prefix frontend run build`
   - Output: `frontend/dist`
   - API: `api/index.py` (Python 3.12)
5. Leave `VITE_API_URL` UNSET. An old localhost value must be removed before building.
6. Deploy. Initial requests can take longer while the scientific libraries and model load.

## Verify the deployed app

- `https://YOUR-APP.vercel.app/` loads CardioLens.
- `https://YOUR-APP.vercel.app/api/health` returns `status: healthy`.
- `https://YOUR-APP.vercel.app/api/docs` displays API documentation.
- Upload a clear JPG/JPEG/PNG smaller than 4 MiB. Review the boundary and lead crops.
- Confirm uploads larger than 4 MiB are rejected locally before a request is sent.

If the UI works but API calls return 404, check the project root and ensure the
API rewrite remains before the frontend fallback. If the backend returns 500,
inspect the Python function logs. Do not expose diagnostic logs in the UI.

## Bundle size

Python functions have a 500 MB uncompressed bundle limit. Linux CPython 3.12
wheels were downloaded for a preflight estimate. Pandas was removed from runtime:
PCA receives the same float64 feature array in the same order. This preserves its
numerical output while reducing the dependency bundle. The estimate is not a
successful Vercel build: Vercel's final build output is authoritative.

The function explicitly includes models, preprocessing and backend code, while
excluding frontend source, tests, editor settings and Python bytecode. The frontend
is built separately as static assets. Model weights and thresholds remain unchanged.

## Local development

The original two-terminal commands still work (frontend port 5173, backend 8000).
The development frontend defaults to localhost:8000. A production build defaults
to `/api`. To test the complete single-project routing locally with Vercel CLI,
run `vercel dev` from this folder after signing in and linking the project.

## References

- https://vercel.com/docs/functions/runtimes/python
- https://vercel.com/docs/functions/limitations
- https://vercel.com/docs/project-configuration/vercel-json
