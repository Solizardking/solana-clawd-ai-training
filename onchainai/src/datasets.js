import Papa from 'papaparse';
import { parse as parseYaml } from 'yaml';
import * as pdfjs from 'pdfjs-dist';
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';
export const secretPattern = /hf_[A-Za-z0-9]{24,}|(?:sk-|nvapi-)[A-Za-z0-9_-]{20,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|(?:api_key|private_key|secret_key)\s*[=:]\s*["']?[A-Za-z0-9_-]{20,}/i;
export async function digest(value) {
  const bytes = typeof value === 'string' ? new TextEncoder().encode(value) : value;
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
}
export function normalizeRow(row) {
  if (Array.isArray(row.messages)) {
    if (row.messages.length && row.messages.every(m => ['system', 'user', 'assistant'].includes(m.role) && typeof m.content === 'string' && m.content.trim())) return { messages: row.messages.map(m => ({ role: m.role, content: m.content })) };
    throw new Error('Invalid messages row');
  }
  const prompt = row.instruction || row.prompt || row.question;
  const answer = row.output || row.response || row.answer || row.completion;
  if (typeof prompt === 'string' && typeof answer === 'string' && prompt.trim() && answer.trim()) return { messages: [{ role: 'user', content: prompt + (row.input ? '\n' + row.input : '') }, { role: 'assistant', content: answer }] };
  throw new Error('Row needs messages or a prompt/answer pair');
}
function textRows(text, source) {
  return text.match(/[\s\S]{1,4500}/g)?.map(chunk => ({ messages: [{ role: 'user', content: `Read the source excerpt from ${source}.` }, { role: 'assistant', content: chunk.trim() }] })).filter(r => r.messages[1].content) || [];
}
export async function buildDataset(files, name, wallet) {
  const rows = []; const sources = []; const warnings = [];
  if (!files.length) throw new Error('Choose at least one file');
  if (files.reduce((sum, f) => sum + f.size, 0) > 15000000) throw new Error('Select at most 15 MB of source files');
  for (const file of files) {
    const extension = file.name.split('.').pop().toLowerCase();
    const bytes = new Uint8Array(await file.arrayBuffer());
    let text = new TextDecoder().decode(bytes); let records = [];
    if (extension === 'pdf') {
      const doc = await pdfjs.getDocument({ data: bytes.slice() }).promise;
      if (doc.numPages > 200) throw new Error('PDF exceeds the 200 page limit');
      text = '';
      for (let i = 1; i <= doc.numPages; i++) text += (await (await doc.getPage(i)).getTextContent()).items.map(item => item.str || '').join(' ') + '\n';
      await doc.destroy();
      if (!text.trim()) throw new Error('PDF has no extractable text; OCR is required for scanned pages');
      records = textRows(text, file.name); warnings.push(`${file.name}: source-reproduction examples; curate questions before training.`);
    } else if (['jsonl', 'ndjson'].includes(extension)) records = text.trim().split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
    else if (extension === 'json') { const data = JSON.parse(text); records = Array.isArray(data) ? data : data.examples || data.data || [data]; }
    else if (extension === 'csv') { const data = Papa.parse(text, { header: true, skipEmptyLines: true }); if (data.errors.length) throw new Error(`CSV: ${data.errors[0].message}`); records = data.data; }
    else if (['yaml', 'yml'].includes(extension)) { const data = parseYaml(text); records = Array.isArray(data) ? data : data.examples || [data]; }
    else if (extension === 'ipynb') { const notebook = JSON.parse(text); text = (notebook.cells || []).map(c => Array.isArray(c.source) ? c.source.join('') : c.source || '').join('\n\n'); records = textRows(text, file.name); warnings.push(`${file.name}: source-reproduction examples; curate before training.`); }
    else if (['txt', 'md'].includes(extension)) { records = textRows(text, file.name); warnings.push(`${file.name}: source-reproduction examples; curate before training.`); }
    else throw new Error(`Unsupported file: ${file.name}`);
    if (secretPattern.test(text) || secretPattern.test(JSON.stringify(records))) throw new Error(`Likely secret detected in ${file.name}; remove credentials before continuing`);
    for (const record of records) rows.push(normalizeRow(record));
    sources.push({ name: file.name, bytes: file.size, sha256: await digest(bytes), examples: records.length });
  }
  const unique = [...new Map(rows.map(row => [JSON.stringify(row), row])).values()];
  if (!unique.length) throw new Error('No valid examples found');
  const train_jsonl = unique.map(r => JSON.stringify(r)).join('\n') + '\n';
  const manifest = { schema: 'clawd.dataset.v1', name, wallet, created_at: new Date().toISOString(), sources, example_count: unique.length, duplicates_removed: rows.length - unique.length, dataset_sha256: await digest(train_jsonl), warnings, quality: { kind: 'structural-check', valid_rows: unique.length, tier: warnings.length ? 'needs-curation' : 'valid-structure', model_quality_evaluated: false } };
  return { train_jsonl, manifest, manifest_sha256: await digest(JSON.stringify(manifest)), preview: unique.slice(0, 3) };
}
