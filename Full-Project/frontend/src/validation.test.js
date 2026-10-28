import test from 'node:test';
import assert from 'node:assert/strict';
import { validateFile, readResponse, validatePrediction } from './validation.js';
const png = [137,80,78,71,13,10,26,10];
const upload = (bytes, name='scan.png', type='image/png') => new File([new Uint8Array(bytes)], name, {type});
test('upload gate accepts JPG, JPEG and PNG and safely forwards disguised WebP to decoder', async () => {
  await validateFile(upload(png));
  await validateFile(upload([255,216,255], 'scan.jpg', 'image/jpeg'));
  await validateFile(upload([255,216,255], 'scan.JPEG', ''));
  await validateFile(upload([82,73,70,70,0,0,0,0,87,69,66,80], 'download.jpg', 'image/webp'));
});
test('empty, oversized, unsupported and forged uploads are blocked', async () => {
  await assert.rejects(validateFile(upload([])), /empty/);
  await assert.rejects(validateFile(upload(new Uint8Array(4*1024*1024+1))), /4 MB/);
  await assert.rejects(validateFile(upload(png,'image.svg','image/svg+xml')), /Unsupported/);
  await assert.rejects(validateFile(upload([1,2,3])), /readable/);
});
test('invalid API payloads cannot become classifications', async () => {
  const valid={prediction:'NORMAL',confidence:.7,probabilities:{NORMAL:.7,AHB:.1,MI:.1,PMI:.1}};
  validatePrediction(valid);
  for (const data of [null,{}, {...valid,prediction:'OTHER'}, {...valid,confidence:NaN}, {...valid,confidence:2}, {...valid,probabilities:{}}]) assert.throws(() => validatePrediction(data), /invalid classification/);
  await assert.rejects(readResponse(new Response('bad')), /unreadable/);
  await assert.rejects(readResponse(Response.json(null)), /unexpected/);
  await assert.rejects(readResponse(Response.json({detail:[{msg:'missing'}]},{status:422})), /could not be processed/);
  await assert.rejects(readResponse(Response.json({detail:'ECG rejected'},{status:422})), /ECG rejected/);
  assert.deepEqual(await readResponse(Response.json(valid)),valid);
});
