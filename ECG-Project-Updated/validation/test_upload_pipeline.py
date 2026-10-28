"""API integration checks using explicitly supplied local samples.

Run: python validation/test_upload_pipeline.py --samples-dir <folder>
No images are changed or copied into fixtures.
"""
from pathlib import Path
import argparse
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx


def run_checks(samples_dir, base_url):
    with httpx.Client(base_url=base_url, timeout=120) as client:
        assert client.get('/health').json()['status'] == 'healthy'
        for filename, rotation in [('risk fellow.jpg', 90), ('unknown.jpg', 270)]:
            raw = (samples_dir / filename).read_bytes()
            response = client.post('/validate?rotation=0', files={'file': (filename, raw, 'image/jpeg')})
            assert response.status_code == 200, response.text[:300]
            result = response.json()
            assert result['valid'], (filename, result['code'])
            prep = result['preprocessing']
            assert prep['rotation_ccw'] == rotation, (filename, prep['rotation_ccw'])
            assert prep['lead_count'] == len(prep['leads']) == 12
            assert len({lead['name'] for lead in prep['leads']}) == 12
            assert all(lead['image'].startswith('data:image/jpeg;base64,') for lead in prep['leads'])
            w, h = prep['original_size']
            x0, y0, x1, y1 = prep['box']
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            assert (x0 > w * .48 if filename.startswith('risk') else x1 < w * .51)
            print('PASS', filename, 'panel isolated, oriented, 12 leads extracted and validated', flush=True)
        image = cv2.imdecode(np.frombuffer((samples_dir / 'unknown.jpg').read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        ok, png = cv2.imencode('.png', image)
        assert ok
        files = {'file': ('scan.png', png.tobytes(), 'image/png')}
        response = client.post('/validate', files=files)
        assert response.status_code == 200 and response.json()['valid']
        print('PASS PNG upload', flush=True)
        response = client.post('/validate?rotation=180', files=files)
        assert response.status_code == 200 and response.json()['twelve_lead']
        assert response.json()['preprocessing']['rotation_ccw'] == 90
        print('PASS manual orientation correction', flush=True)
        for data in (b'', b'<svg></svg>', b'%PDF-1.4'):
            response = client.post('/predict', files={'file': ('bad.jpg', data, 'image/jpeg')})
            assert response.status_code == 422 and 'prediction' not in response.json()
        assert client.post('/validate?rotation=90', files=files).status_code == 422
        response = client.post('/predict?rotation=0', files=files)
        assert response.status_code in (200, 422)
        result = response.json()
        if response.status_code == 200:
            assert result['prediction'] in ('NORMAL', 'AHB', 'MI', 'PMI')
            assert 0 <= result['confidence'] <= 1
        else:
            assert result['code'] == 'LOW_CONFIDENCE' and 'prediction' not in result
        print('PASS invalid uploads, rotation parameter and prediction confidence gate', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    run_checks(args.samples_dir, args.base_url)
