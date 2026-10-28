import { useEffect, useRef, useState } from 'react';
import { Metric, Note, BarRow } from '../components/ui.jsx';
import { CLASS_INFO } from '../data.js';
import { validateFile, validatePrediction, readResponse } from '../validation.js';
const checkName = name => name.replaceAll('_', ' ');
export default function Analysis({ apiUrl, status, palette }) {

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [validation, setValidation] = useState(null);
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState('idle');
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [reviewed, setReviewed] = useState(false);
  const [previewMode, setPreviewMode] = useState('boundary');
  const input = useRef(null);
  const request = useRef(null);
  const busy = phase === 'validating' || phase === 'analyzing';

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => () => request.current?.abort(), []);

  function reset() {
    request.current?.abort();
    request.current = null;
    setFile(null); setPreview(''); setValidation(null); setResult(null); setError(''); setPhase('idle');
    setRotation(0); setReviewed(false); setPreviewMode('boundary');
    if (input.current) input.current.value = '';
  }

  async function selectFile(selected, correction = 0) {
    reset();
    if (!selected) return;
    const controller = new AbortController();
    setRotation(correction);
    request.current = controller;
    setPhase('validating');
    try {
      await validateFile(selected);
      if (controller.signal.aborted) return;
      setFile(selected); setPreview(URL.createObjectURL(selected));
      const body = new FormData(); body.append('file', selected);
      const response = await fetch(`${apiUrl}/validate?rotation=${correction}`, { method: 'POST', body, signal: controller.signal });
      const data = await readResponse(response);
      if (typeof data.valid !== 'boolean' || !Array.isArray(data.checks)) throw new Error('Invalid validation response. Please try again.');
      setValidation(data);
      if (!data.valid) throw new Error(data.message || 'This image did not pass ECG validation.');
      setPhase('ready');
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof TypeError ? 'Cannot reach the analysis server. Check that the backend is running, then select the image again.' : err.message);
      setPhase('error');
    }
  }

  async function analyze() {
    if (!file || !validation?.valid || !reviewed || busy) return;
    const controller = new AbortController(); request.current = controller;
    setPhase('analyzing'); setError(''); setResult(null);
    try {
      const body = new FormData(); body.append('file', file);
      const response = await fetch(`${apiUrl}/predict?rotation=${rotation}`, { method: 'POST', body, signal: controller.signal });
      const data = await readResponse(response);
      validatePrediction(data);
      setResult(data); setPhase('complete');
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof TypeError ? 'Cannot reach the analysis server. Please try again.' : err.message);
      setPhase('ready');
    }
  }

  const prep = validation?.preprocessing;
  const displayPreview = prep ? previewMode === 'boundary' ? prep.annotated_image : previewMode === 'region' ? prep.upright_image : preview : preview;

  const info = result ? CLASS_INFO[result.prediction] : null;
  function downloadCsv() {
    const quote = v => `"${String(v).replaceAll('"','""')}"`;
    const csv = 'file,predicted_class,label,confidence\r\n' + [file.name,result.prediction,info.title,result.confidence].map(quote).join(',');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='ecg_analysis_result.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  return <>
    <div className="grid3"><Metric label="INPUT" value="Image" sub="JPG, JPEG or PNG · max 4 MB" /><Metric label="EXTRACTION" value={prep ? `${prep.lead_count} leads` : '12-lead'} sub={prep ? `${prep.layout} layout · ${prep.rotation_ccw}° correction` : 'Locate → orient → extract → validate'} tone="m2" /><Metric label="OUTPUT" value="Class + probabilities" sub="NORMAL · AHB · MI · PMI" tone="m3" /></div>
    <h2 className="section-title gap">Start a new analysis</h2><p className="subtle">Upload a complete ECG page. We isolate the ECG from adjacent notes, correct its orientation, extract twelve leads and validate before classification.</p>
    {status==='offline' && <div className="warn">The analysis server is unavailable. Connection status updates automatically.</div>}
    <div className={`drop${dragging?' dragging':''}`} onDragOver={e=>{e.preventDefault();if(!busy)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);if(!busy){if(e.dataTransfer.files.length!==1){reset();setError('Please upload one ECG image at a time.');}else selectFile(e.dataTransfer.files[0]);}}}>
      <input ref={input} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" hidden disabled={busy} onChange={e=>selectFile(e.target.files[0])}/><span className="drop-ic" aria-hidden="true">↑</span><b>{file?'Choose a different ECG image':'Choose an ECG image'}</b><span className="subtle">Drop your file here, or browse your device</span><button className="btn" disabled={busy} onClick={()=>input.current?.click()}>Browse files</button><small className="caption">JPG / JPEG / PNG · up to 4 MB · minimum 300 × 300 px</small>
    </div>
    <div aria-live="polite">{busy&&<p className="analysis-activity">{phase==='validating'?'Locating ECG region, correcting orientation, extracting leads and validating…':'Running the trained classifier…'}</p>}</div>
    {file&&<div className="grid-preview"><section className="analysis-preview"><h2 className="section-title">{prep?'Detected ECG region':'Image preview'}</h2><p className="subtle">{prep?'The green boundary marks the region used for extraction.':'The server is checking your image.'}</p><img className="preview" src={displayPreview} alt={prep&&previewMode==='boundary'?'Original image with detected ECG boundary':previewMode==='region'?'Upright ECG region':'Original upload'}/>{prep&&<><div className="pills" aria-label="Preview mode">{[['boundary','ECG boundary'],['region','Upright region'],['original','Original']].map(([mode,label])=><button key={mode} className={`pill${previewMode===mode?' on':''}`} aria-pressed={previewMode===mode} onClick={()=>setPreviewMode(mode)}>{label}</button>)}</div><button className="btn" disabled={busy} onClick={()=>selectFile(file,rotation===0?180:0)}>Rotate ECG 180° & recheck ↻</button></>}<p className="caption">{file.name} · {(file.size/1048576).toFixed(2)} MB</p></section>
    <section><h2 className="section-title">Validation checklist</h2><p className="subtle">Live results from the ECG validation pipeline.</p>{validation?<><div className={`validation-banner ${validation.valid?'pass':'fail'}`}>{validation.valid?'12-lead ECG validated':validation.twelve_lead?'ECG detected · model validation failed':'Image validation failed'}</div><p className="caption">{validation.checks.filter(c=>c.passed).length} of {validation.checks.length} checks passed</p><ul className="validation-checks">{validation.checks.map((c,i)=><li key={`${c.name}-${i}`}><span>{checkName(c.name)}</span><strong className={c.passed?'check-pass':'check-fail'}>{c.passed?'PASS':'FAIL'}</strong></li>)}</ul></>:<div className="panel gap"><p className="subtle">{busy?'Checks are in progress…':'Validation results will appear here.'}</p></div>}{validation?.valid&&<label className="check-row review-row"><input type="checkbox" checked={reviewed} onChange={e=>setReviewed(e.target.checked)}/><span>I reviewed the orientation and all 12 lead crops below.</span></label>}<div className="btn-row"><button className="btn primary" disabled={busy||!validation?.valid||!reviewed} onClick={analyze}>{phase==='analyzing'?'Analysing…':'Run ECG analysis →'}</button><button className="btn" onClick={reset}>Clear</button></div></section></div>}
    {error&&<div className="error" role="alert">{error}</div>}
    {prep&&<section className="gap"><h2 className="section-title">Explore the 12 extracted leads</h2><p className="subtle">{prep.label_method}</p><div className="lead-gallery">{prep.leads.map(lead=><figure key={lead.name}><figcaption><strong>{lead.name}</strong><small>EXTRACTED REGION</small></figcaption><img src={lead.image} alt={`Extracted region assigned to lead ${lead.name}; verify printed label`} loading="lazy"/></figure>)}</div>{validation.warnings?.map((w,i)=><div className="warn" key={i}>{w}</div>)}</section>}
    {result&&<section className="results"><hr/><h2 className="section-title">Analysis results</h2><div className="result"><div className="result-label">MODEL CLASSIFICATION · {file.name}</div><div className="result-value">{info.title}</div><p>{info.description}</p><span className="rchip">Class: {result.prediction}</span><span className="rchip">Confidence estimate: {(result.confidence*100).toFixed(2)}%</span></div><h3 className="section-title gap">Model-reported class probabilities</h3>{Object.entries(result.probabilities).sort((a,b)=>b[1]-a[1]).map(([code,p])=><BarRow key={code} label={code} value={p} max={1} color={code===result.prediction?palette[0]:'#b6c2d2'} text={`${(p*100).toFixed(1)}%`}/>)}<p className="caption">Model probabilities are estimates, not diagnostic certainty.</p>{result.warnings?.filter(w=>!validation.warnings?.includes(w)).map((w,i)=><div className="warn" key={i}>{w}</div>)}<div className="btn-row"><button className="btn" onClick={downloadCsv}>Download result CSV</button><button className="btn" onClick={reset}>Start a new analysis</button></div><Note>This classification is a research output, not a medical diagnosis.</Note></section>}
    {!file&&!busy&&<div className="panel gap"><h2 className="section-title">Waiting for an ECG image</h2><p className="subtle">Upload an image to inspect the ECG boundary, validation checks and twelve lead crops. Open About the Project to explore the project overview.</p></div>}
  </>;
}
