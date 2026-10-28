"""ASGI routing, upload limits and PCA equivalence checks (development only)."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from api.index import app
from backend.main import pca
from preprocessing.ecg_validator import transform_features
import pandas as pd  # Optional development-only comparison dependency.

values=np.random.default_rng(42).random((3,3060))
expected=pca.transform(pd.DataFrame(values,columns=pca.feature_names_in_))
np.testing.assert_allclose(transform_features(pca,values),expected,rtol=0,atol=0)
print('PASS PCA output is bit-identical without DataFrame input',flush=True)
with TestClient(app) as client:
    assert client.get('/api/health').json()['status']=='healthy'
    assert client.get('/api/docs').status_code==200
    assert client.get('/api/openapi.json').status_code==200
    for content in [b'',b'<svg></svg>']:
        r=client.post('/api/predict',files={'file':('bad.jpg',content,'image/jpeg')})
        assert r.status_code==422 and 'prediction' not in r.json()
    r=client.post('/api/validate',files={'file':('large.jpg',b'x'*(4*1024*1024+1),'image/jpeg')})
    assert r.status_code==200 and r.json()['code']=='FILE_TOO_LARGE'
    assert client.post('/api/validate?rotation=90',files={'file':('bad.jpg',b'','image/jpeg')}).status_code==422
    print('PASS /api routing, docs, validation, upload cap and rotation query',flush=True)
