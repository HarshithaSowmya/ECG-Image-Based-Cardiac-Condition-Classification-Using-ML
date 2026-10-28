import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const config=JSON.parse(readFileSync(new URL('../vercel.json',import.meta.url)));
test('Vercel build includes one ASGI entry point and model files',()=>{
 assert.equal(config.outputDirectory,'frontend/dist');
 assert.equal(config.functions['api/index.py'].maxDuration,300);
 assert.ok(config.functions['api/index.py'].includeFiles.includes('models/**'));
 assert.equal(config.rewrites[0].source,'/api/:path*');
 assert.equal(config.rewrites[0].destination,'/api/index');
});
