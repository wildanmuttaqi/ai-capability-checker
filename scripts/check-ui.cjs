// Run: node scripts/check-ui.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../ui/index.html'), 'utf8');
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const source = html.slice(html.indexOf('function conclusion(r){'), html.indexOf('const originalReport=report;'));
const context = vm.createContext({number: n => n.toFixed(2)});
vm.runInContext(source, context);
const report = {benchmark:{generation_tps:3.49,backend:'CPU'},recommendation:{models:[
    {parameters_b:4,measured_tps:3.49},
    {parameters_b:8,memory:{status:'candidate'}},
    {parameters_b:14,memory:{status:'memory pressure'}}
]}};
const text = context.conclusion(report).join('\n');
assert.match(text,/8B Q4_K_M/);
assert.match(text,/hindari 14B/);
assert.match(text,/kurang nyaman/);
assert.match(text,/belum menyediakan benchmark pembanding/);
assert.doesNotMatch(text,/M3 Pro|4070/);
console.log('UI conclusions: OK');