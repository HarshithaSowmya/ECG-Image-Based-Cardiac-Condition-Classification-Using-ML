import { useState } from "react";
import { Note } from "../components/ui.jsx";
import { FUTURE_SCOPE } from "../data.js";

export default function Limits() {
  const [checked, setChecked] = useState({});
  return (
    <>
      <div className="panel">
        <div className="section-title">Research prototype, not a medical device</div>
        <p className="subtle big">This project classifies a particular ECG image representation using trained PCA and classifier
          artifacts. It is not a diagnosis, cannot rule out heart disease, and must not be used to make treatment decisions.</p>
      </div>
      <div className="grid2">
        <div>
          <div className="panel">
            <div className="section-title">Known limitations</div>
            <ul>
              <li>Limited dataset size and diversity</li>
              <li>ECG image quality affects waveform extraction</li>
              <li>Contour extraction may fail on noisy or unclear images</li>
              <li>Trained only on four predefined classes</li>
              <li>External patient-level validation is required</li>
            </ul>
          </div>
          <div className="panel">
            <div className="section-title">Compatibility</div>
            <p className="subtle">The pipeline isolates supported ECG panels, corrects quarter-turn orientation and extracts
              standard 3×4, 6×2 or 12×1 layouts before validation. Review the region and lead labels: heavy skew,
              overlapping traces, single-lead strips and unfamiliar layouts may still fail.</p>
            <div className="section-title">Model files</div>
            <p className="subtle">Pickled scikit-learn models are version-sensitive. If loading fails, install the pinned
              scikit-learn version. A loading failure is not an ECG result.</p>
          </div>
        </div>
        <div>
          <div className="section-title">Future scope</div>
          {FUTURE_SCOPE.map((f) => (
            <label className="check-row" key={f}>
              <input type="checkbox" checked={!!checked[f]} onChange={(e) => setChecked({ ...checked, [f]: e.target.checked })} />
              <span>{f}</span>
            </label>
          ))}
          <p className="caption">Roadmap: data → methods → deployment → clinical validation. (Tick items to explore; nothing is saved.)</p>
        </div>
      </div>
      <Note>If you have symptoms or concerns, consult a qualified healthcare professional. Seek emergency care for possible
        heart-attack symptoms.</Note>
    </>
  );
}
