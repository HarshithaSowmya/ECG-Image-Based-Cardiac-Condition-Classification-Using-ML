import { useEffect, useState } from "react";
import "./App.css";
import Hero from "./components/Hero.jsx";
import Analysis from "./pages/Analysis.jsx";
import About from "./pages/About.jsx";
import Limits from "./pages/Limits.jsx";
import { THEMES } from "./data.js";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const PAGES = [
  { id: "analysis", icon: "🫀", label: "ECG Analysis" },
  { id: "about", icon: "📘", label: "About the Project" },
  { id: "limits", icon: "⚠️", label: "Limits & Responsible Use" },
];

const HERO = {
  analysis: ["Understand the signal.\nExplore the pattern.", "Upload a supported 12-lead ECG image and run the project's trained preprocessing and classification pipeline, with every step visible."],
  about: ["About the project", "ECG Image-Based Cardiac Condition Classification Using ML: from a photographed or scanned ECG page to a four-class prediction, using a lightweight classical-ML pipeline."],
  limits: ["Limits & responsible use", "What this prototype can and cannot do, and why its output must never replace a clinician."],
};

const STATUS_TEXT = { checking: "Checking server…", online: "Model Online", degraded: "Model Not Loaded", offline: "Server Offline" };

export default function App() {
  const [page, setPage] = useState("analysis");
  const [theme, setTheme] = useState("Coral Pulse");
  const [bpm, setBpm] = useState(72);
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        if (!cancelled) setStatus((data.status === "healthy" || data.model_ready) ? "online" : "degraded");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };
    check();
    const t = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  const palette = THEMES[theme];
  const vars = { "--a1": palette[0], "--a2": palette[1], "--a3": palette[2], "--beat": `${(120 / bpm).toFixed(2)}s` };
  const [heroTitle, heroText] = HERO[page];

  return (
    <div className="shell" style={vars}>
      <aside className="sidebar">
        <div className="brand">♥ CardioLens</div>
        <div className="brand-sub">ECG intelligence workspace</div>
        <div className={`status status-${status}`}><span className="status-dot" />{STATUS_TEXT[status]}</div>

        <hr />
        <div className="side-h">NAVIGATE</div>
        <nav>
          {PAGES.map((p) => (
            <button key={p.id} className={`nav${page === p.id ? " on" : ""}`} onClick={() => setPage(p.id)}>
              <span>{p.icon}</span> {p.label}
            </button>
          ))}
        </nav>

        <hr />
        <div className="side-h">MAKE IT YOURS</div>
        <div className="theme-list">
          {Object.keys(THEMES).map((t) => (
            <button key={t} className={`theme${theme === t ? " on" : ""}`} onClick={() => setTheme(t)}>
              <i style={{ background: THEMES[t][0] }} />{t}
            </button>
          ))}
        </div>
        <label className="bpm">
          Heartbeat speed (BPM): <b>{bpm}</b>
          <input type="range" min="40" max="160" value={bpm} onChange={(e) => setBpm(Number(e.target.value))} />
          <small>Only changes the header animation.</small>
        </label>

        <hr />
        <div className="side-h">MODEL PIPELINE</div>
        <p className="side-note">ECG region detection → orientation correction → 12-lead extraction → validation → PCA → SVM classifier</p>
        <p className="side-note">The image is sent only to the backend you run yourself (<code>{API_URL}</code>).</p>
      </aside>

      <div className="backdrop">
        <main className="sheet">
          <Hero title={heroTitle.split("\n").flatMap((l, i, a) => (i < a.length - 1 ? [l, <br key={i} />] : [l]))} text={heroText} />
          <div hidden={page !== "analysis"}><Analysis apiUrl={API_URL} status={status} palette={palette} /></div>
          {page === "about" && <About palette={palette} />}
          {page === "limits" && <Limits />}
          <footer className="foot">CardioLens · ECG image classification research prototype · Not for clinical decision-making</footer>
        </main>
      </div>
    </div>
  );
}
