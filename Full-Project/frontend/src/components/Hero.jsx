const ECG_PATH =
  "M0 40 H70 L80 40 L90 28 L100 40 H130 L140 46 L152 4 L164 66 L176 40 H220 " +
  "Q236 22 252 40 H330 L340 40 L350 28 L360 40 H390 L400 46 L412 4 L424 66 L436 40 H480 " +
  "Q496 22 512 40 H600";

export default function Hero({ title, text }) {
  return (
    <section className="hero">
      <span className="chip">Cardiovascular AI · Research prototype</span>
      <h1>{title}</h1>
      <p>{text}</p>
      <div className="chips">
        <span className="chip">12-lead images</span>
        <span className="chip">PCA + SVM</span>
        <span className="chip">4 cardiac classes</span>
      </div>
      <svg className="ecg" viewBox="0 0 600 70" preserveAspectRatio="none" aria-hidden="true">
        <path pathLength="1000" d={ECG_PATH} />
      </svg>
    </section>
  );
}
