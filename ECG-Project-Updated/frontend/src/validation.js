const MAX_BYTES = 4 * 1024 * 1024;
const LABELS = ['NORMAL', 'AHB', 'MI', 'PMI'];

export async function validateFile(file) {
  if (!file.size) throw new Error('The file is empty. Choose a readable ECG image.');
  if (file.size > MAX_BYTES) throw new Error('The file exceeds 4 MB. Choose a smaller ECG image.');
  if (!/\.(jpe?g|png)$/i.test(file.name)) {
    throw new Error('Unsupported file format. Please choose a JPEG or PNG image.');
  }
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const png = [137, 80, 78, 71, 13, 10, 26, 10].every((b, i) => bytes[i] === b);
  const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
  const webp = String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF' && String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP';
  if (!png && !jpeg && !webp) throw new Error('The file does not contain a readable image. Choose a valid JPEG or PNG.');
  // Full decoding, resolution, ECG layout and signal checks are enforced by the API.
}

export async function readResponse(response) {
  let data;
  try { data = await response.json(); }
  catch { throw new Error('The server returned an unreadable response. Please try again.'); }
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('The server returned an unexpected response. Please try again.');
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be processed. Please try again.');
  return data;
}

export function validatePrediction(data) {
  if (!data || !LABELS.includes(data.prediction) || !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1 ||
      !LABELS.every(key => Number.isFinite(data.probabilities?.[key]) && data.probabilities[key] >= 0 && data.probabilities[key] <= 1)) {
    throw new Error('The server returned an invalid classification result. Please try again.');
  }
}
