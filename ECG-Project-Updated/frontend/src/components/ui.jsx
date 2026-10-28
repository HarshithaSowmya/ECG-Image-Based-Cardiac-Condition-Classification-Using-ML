import { useState } from "react";

export function Metric({ label, value, sub, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="subtle">{sub}</div>
    </div>
  );
}

export function Note({ children }) {
  return <div className="note">{children}</div>;
}

export function Tabs({ tabs }) {
  const [active, setActive] = useState(0);
  return (
    <>
      <div className="tabs" role="tablist">
        {tabs.map((t, i) => (
          <button key={t.label} role="tab" aria-selected={i === active}
            className={`tab${i === active ? " on" : ""}`} onClick={() => setActive(i)}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="tab-body">{tabs[active].content}</div>
    </>
  );
}

// Donut chart drawn with SVG circles (no chart library needed)
export function Donut({ data, colors, size = 260 }) {
  const total = data.reduce((sum, d) => sum + d.value, 0);
  const r = 90, c = 2 * Math.PI * r;
  const arcs = data.map((d, i) => {
    const before = data.slice(0, i).reduce((sum, x) => sum + x.value, 0);
    return { ...d, len: (d.value / total) * c, offset: (before / total) * c, color: colors[i] };
  });
  return (
    <svg viewBox="0 0 260 260" width="100%" style={{ maxWidth: size }} role="img" aria-label="Class distribution">
      <g transform="rotate(-90 130 130)">
        {arcs.map((a) => (
          <circle key={a.label} cx="130" cy="130" r={r} fill="none" stroke={a.color} strokeWidth="46"
            strokeDasharray={`${a.len - 2} ${c - a.len + 2}`} strokeDashoffset={-a.offset}>
            <title>{`${a.label}: ${a.value}`}</title>
          </circle>
        ))}
      </g>
      <text x="130" y="126" textAnchor="middle" className="donut-n">{total}</text>
      <text x="130" y="148" textAnchor="middle" className="donut-l">images</text>
    </svg>
  );
}

export function Legend({ items, colors }) {
  return (
    <div className="legend">
      {items.map((t, i) => (
        <span key={t}><i style={{ background: colors[i] }} />{t}</span>
      ))}
    </div>
  );
}

export function BarRow({ label, value, max = 1, color, text }) {
  return (
    <div className="bar-row">
      <span className="bar-name">{label}</span>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.min(100, (value / max) * 100)}%`, background: color }} /></div>
      <span className="bar-val">{text ?? value}</span>
    </div>
  );
}

// 255-sample waveform, y flipped so ECG peaks point up (same as the Streamlit app)
export function Waveform({ values, color }) {
  const W = 640, H = 240, P = 18;
  const pts = values.map((v, i) => `${(P + (i / (values.length - 1)) * (W - 2 * P)).toFixed(1)},${(P + v * (H - 2 * P)).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="wave" role="img" aria-label="Normalised lead waveform">
      {[0.25, 0.5, 0.75].map((g) => (
        <line key={g} x1={P} x2={W - P} y1={P + g * (H - 2 * P)} y2={P + g * (H - 2 * P)} className="grid" />
      ))}
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
