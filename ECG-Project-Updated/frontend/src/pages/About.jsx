import { useState } from "react";
import { Metric, Note, Tabs, Donut, Legend, BarRow } from "../components/ui.jsx";
import {
  OBJECTIVES, DATASET, STAGES, CLASSIFIERS, CLASSES, REPORT, CONFUSION, TEAM, REFERENCES,
} from "../data.js";

function Overview({ a1 }) {
  return (
    <>
      <div className="panel">
        <div className="section-title">Problem statement</div>
        <p className="subtle big">
          ECG reports hold important information about cardiac activity, but extracting it from ECG <i>images</i> is
          hard because of image noise, background grids and irrelevant pixels. This project processes the image
          automatically, extracts the waveform of every lead, turns it into numbers, and classifies it.
        </p>
      </div>
      <div className="grid4">
        <Metric label="DATASET" value="928" sub="ECG images, 4 classes" />
        <Metric label="FEATURES" value="3060 → 190" sub="12 leads × 255 samples, then PCA" tone="m2" />
        <Metric label="BEST MODEL" value="SVM" sub="Best 5-fold CV score on the training split" tone="m3" />
        <Metric label="TEST ACCURACY" value="93.01%" sub="On a 186-image test set" tone="m4" />
      </div>
      <div className="section-title gap">Objectives</div>
      <div className="grid3">
        {OBJECTIVES.map(([t, d], i) => (
          <div className="obj" key={t}><span className="n">{i + 1}</span><b>{t}</b><span className="subtle">{d}</span></div>
        ))}
      </div>
      <div className="section-title gap">Why this approach? The research gap</div>
      <div className="grid2">
        <div className="panel">
          <ul>
            <li>Limited dataset size and diversity in existing work</li>
            <li>Lack of independent / external validation</li>
            <li>Deep learning is computationally expensive</li>
            <li>Complex preprocessing in some methods; the grid and background interfere with features</li>
            <li>Limited explainability in some models</li>
          </ul>
        </div>
        <div className="stage" style={{ borderLeft: `8px solid ${a1}` }}>
          <h3>Our direction</h3>
          <p className="subtle big">An efficient, lightweight and ECG-signal-focused classification system: classical ML on
            compact waveform features, instead of an end-to-end CNN.</p>
        </div>
      </div>
      <div className="section-title gap">Key contributions</div>
      <ul className="plain">
        <li>12-lead ECG processing from images</li>
        <li>Waveform-based feature extraction</li>
        <li>3060 → 190 feature reduction with PCA</li>
        <li>Classical ML classification</li>
        <li>FastAPI + React deployment</li>
      </ul>
    </>
  );
}

function DatasetTab({ palette }) {
  const [pick, setPick] = useState("NORMAL");
  const total = DATASET.reduce((s, d) => s + d.images, 0);
  const row = DATASET.find((d) => d.cls === pick);
  const colors = [palette[0], palette[2], palette[1], "#12b5a6"];
  return (
    <>
      <div className="section-title">Four-class ECG cardiac-condition dataset</div>
      <p className="subtle">928 ECG images · 2213 × 1572 px · 12-lead information. Pick a class to see its share.</p>
      <div className="grid2">
        <div className="center">
          <Donut data={DATASET.map((d) => ({ label: d.cls, value: d.images }))} colors={colors} />
          <Legend items={DATASET.map((d) => `${d.cls} (${d.images})`)} colors={colors} />
        </div>
        <div>
          <div className="pills">
            {DATASET.map((d) => (
              <button key={d.cls} className={`pill${pick === d.cls ? " on" : ""}`} onClick={() => setPick(d.cls)}>{d.cls}</button>
            ))}
          </div>
          <div className="stage" style={{ borderLeft: `8px solid ${palette[0]}` }}>
            <h3>{row.cls} · {row.meaning}</h3>
            <p className="subtle big">{row.images} images, {((row.images / total) * 100).toFixed(1)}% of the dataset.</p>
          </div>
          <table className="table">
            <thead><tr><th>Class</th><th>Images</th><th>Meaning</th></tr></thead>
            <tbody>{DATASET.map((d) => (<tr key={d.cls}><td>{d.cls}</td><td>{d.images}</td><td>{d.meaning}</td></tr>))}</tbody>
          </table>
          <p className="caption">Dataset: Kaggle “ECG Image” by kanishkarathore1604.</p>
        </div>
      </div>
    </>
  );
}

function PipelineTab({ palette }) {
  const [idx, setIdx] = useState(0);
  const [title, sub, desc] = STAGES[idx];
  return (
    <>
      <div className="section-title">From ECG image to class label</div>
      <p className="subtle">Pick a stage to see what happens to the image.</p>
      <div className="stepper">
        {STAGES.map((s, i) => (
          <button key={s[0]} className={`step${i === idx ? " on" : ""}${i < idx ? " done" : ""}`} onClick={() => setIdx(i)} title={s[0]}>
            <span>{i + 1}</span>
          </button>
        ))}
      </div>
      <div className="progress"><div className="progress-fill" style={{ width: `${((idx + 1) / STAGES.length) * 100}%` }} /></div>
      <div className="stage" style={{ borderLeft: `8px solid ${palette[idx % palette.length]}` }}>
        <h3>{idx + 1}. {title}</h3>
        <b className="accent">{sub}</b>
        <p className="subtle big">{desc}</p>
      </div>
      <div className="section-title gap">Classifiers compared</div>
      <div className="grid4">
        {CLASSIFIERS.map(([n, d]) => (<div className="obj short" key={n}><b>{n}</b><span className="subtle">{d}</span></div>))}
      </div>
      <div className="section-title gap">Deployed architecture</div>
      <div className="grid3">
        <div className="panel"><b>Frontend · React</b><ul><li>ECG image upload</li><li>Result view</li></ul></div>
        <div className="panel"><b>Backend · FastAPI</b><ul><li>Image preprocessing</li><li>Lead extraction</li><li>Feature extraction</li></ul></div>
        <div className="panel"><b>ML · PCA + SVM</b><ul><li>PCA transformation</li><li>SVM prediction</li></ul></div>
      </div>
    </>
  );
}

function ResultsTab({ palette }) {
  const [metric, setMetric] = useState("Precision");
  const colors = [palette[0], palette[2], palette[1], "#12b5a6"];
  const max = Math.max(...CONFUSION.flat());
  return (
    <>
      <div className="section-title">Final SVM performance</div>
      <p className="subtle">Project evaluation figures supplied with this interface (186 test images). These figures are not recomputed when you upload an ECG.</p>
      <div className="grid4">
        <Metric label="ACCURACY" value="93.01%" sub="Overall" />
        <Metric label="PRECISION" value="93.00%" sub="Macro" tone="m2" />
        <Metric label="RECALL" value="92.46%" sub="Macro" tone="m3" />
        <Metric label="MACRO F1" value="92.67%" sub="Macro" tone="m4" />
      </div>
      <div className="grid2 gap">
        <div>
          <div className="section-title">Per-class report</div>
          <div className="pills">
            {["Precision", "Recall", "F1-score"].map((m) => (
              <button key={m} className={`pill${metric === m ? " on" : ""}`} onClick={() => setMetric(m)}>{m}</button>
            ))}
          </div>
          {REPORT.map((r, i) => (
            <BarRow key={r.cls} label={r.cls} value={r[metric]} max={1} color={colors[i]} text={r[metric].toFixed(2)} />
          ))}
          <table className="table">
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>{REPORT.map((r) => (
              <tr key={r.cls}><td>{r.cls}</td><td>{r.Precision.toFixed(2)}</td><td>{r.Recall.toFixed(2)}</td><td>{r["F1-score"].toFixed(2)}</td><td>{r.Support}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <div>
          <div className="section-title">Confusion matrix</div>
          <div className="cm">
            <span />
            {CLASSES.map((c) => (<span key={c} className="cm-h">{c}</span>))}
            {CONFUSION.map((row, i) => (
              <div className="cm-row" key={CLASSES[i]}>
                <span className="cm-h">{CLASSES[i]}</span>
                {row.map((v, j) => (
                  <span key={j} className="cm-c" style={{ background: `color-mix(in srgb, var(--a3) ${Math.round((v / max) * 100)}%, #f4f1ff)`, color: v / max > 0.4 ? "#fff" : "#1b1d3a" }}>{v}</span>
                ))}
              </div>
            ))}
          </div>
          <p className="caption">Rows are the true class, columns the predicted class. MI is separated perfectly; most errors are AHB ↔ NORMAL/PMI.</p>
        </div>
      </div>
      <Note><b>Read these numbers carefully:</b> they come from one project dataset and one evaluation protocol, so they should
        not be compared directly with accuracies in published papers. External, patient-level validation is still required.</Note>
    </>
  );
}

function TeamTab() {
  return (
    <>
      <div className="section-title">Team</div>
      <p className="subtle">Under the guidance of <b>Dr. Chattu Kannaiah Sir</b></p>
      <div className="grid4">
        {TEAM.map((n) => (
          <div className="person" key={n}>
            <div className="avatar">{n.replace(/\./g, " ").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]).join("")}</div>
            <b>{n}</b>
          </div>
        ))}
      </div>
      <div className="section-title gap">References</div>
      <ol className="refs">{REFERENCES.map((r) => (<li key={r}>{r}</li>))}</ol>
    </>
  );
}

export default function About({ palette }) {
  return (
    <Tabs tabs={[
      { label: "Overview", content: <Overview a1={palette[0]} /> },
      { label: "Dataset", content: <DatasetTab palette={palette} /> },
      { label: "Pipeline", content: <PipelineTab palette={palette} /> },
      { label: "Results", content: <ResultsTab palette={palette} /> },
      { label: "Team & references", content: <TeamTab /> },
    ]} />
  );
}
