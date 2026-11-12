import test from 'node:test';
import assert from 'node:assert/strict';
import { hubCatalog, resetHubCache } from '../server/service.mjs';

function item(id, extras = {}) {
  return { id, downloads: 10, downloadsAllTime: 100, likes: 2, sha: 'abc', private: false, siblings: [], cardData: {}, lastModified: '2026-11-01T00:00:00.000Z', ...extras };
}

function jsonResponse(data, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => data };
}

function installFetch(handler) {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    calls.push({ url: String(url), options });
    return handler(String(url), options, calls);
  };
  return { calls, restore() { globalThis.fetch = original; } };
}

function restoreEnv(name, previous) {
  if (previous === undefined) delete process.env[name];
  else process.env[name] = previous;
}

test('hub catalog totals live public downloads and skips private repos', async () => {
  resetHubCache();
  const previousToken = process.env.HF_TOKEN;
  delete process.env.HF_TOKEN;
  const stub = installFetch((url) => {
    if (url.includes('whoami')) throw new Error('whoami should not be called without HF_TOKEN');
    if (url.includes('/api/models') && url.includes('solanaclawd')) {
      return jsonResponse([
        item('solanaclawd/core-ai', { downloads: 5, downloadsAllTime: 50, likes: 1 }),
        item('solanaclawd/secret', { private: true, downloads: 999, downloadsAllTime: 999, likes: 50 }),
      ]);
    }
    if (url.includes('/api/models') && url.includes('ordlibrary')) {
      return jsonResponse([item('ordlibrary/ordgpt', { downloads: 3, downloadsAllTime: 7, likes: 4 })]);
    }
    if (url.includes('/api/datasets')) return jsonResponse([item('solanaclawd/data', { downloads: 2, downloadsAllTime: 8, likes: 1 })]);
    if (url.includes('/api/spaces')) return jsonResponse([]);
    throw new Error(`unexpected ${url}`);
  });
  try {
    const catalog = await hubCatalog();
    assert.equal(catalog.live, true);
    assert.deepEqual(catalog.models.map(model => model.id).sort(), ['ordlibrary/ordgpt', 'solanaclawd/core-ai']);
    assert.equal(catalog.stats.model_downloads_30d, 8);
    assert.equal(catalog.stats.model_downloads_all_time, 57);
    assert.equal(catalog.stats.dataset_downloads_all_time, 8);
    assert.equal(catalog.stats.likes, 6);
    assert.equal(catalog.stats.authenticated, false);
    assert.equal(JSON.stringify(catalog).includes('hf_'), false);
    assert.ok(stub.calls.every(call => call.url.includes('expand[]=downloadsAllTime')));
    assert.equal(await hubCatalog(), catalog);
  } finally {
    stub.restore();
    restoreEnv('HF_TOKEN', previousToken);
  }
});

test('HF_TOKEN includes the signed-in account and orgs without exposing the token', async () => {
  resetHubCache();
  const previousToken = process.env.HF_TOKEN;
  const previousAuthors = process.env.HF_AUTHORS;
  process.env.HF_TOKEN = 'hf_live_fixture_token_do_not_leak';
  delete process.env.HF_AUTHORS;
  const stub = installFetch((url, options) => {
    assert.equal(options.headers?.Authorization, 'Bearer hf_live_fixture_token_do_not_leak');
    if (url.includes('whoami-v2')) return jsonResponse({ name: 'solizardking', orgs: [{ name: 'themusebook' }] });
    if (url.includes('author=themusebook') && url.includes('/api/models')) {
      return jsonResponse([item('themusebook/alpha', { downloads: 1, downloadsAllTime: 11, likes: 0 })]);
    }
    if (url.includes('/api/models') || url.includes('/api/datasets') || url.includes('/api/spaces')) return jsonResponse([]);
    throw new Error(`unexpected ${url}`);
  });
  try {
    const catalog = await hubCatalog();
    assert.equal(catalog.stats.authenticated, true);
    assert.equal(catalog.stats.account, 'solizardking');
    assert.ok(catalog.stats.authors.includes('solizardking'));
    assert.ok(catalog.stats.authors.includes('themusebook'));
    assert.ok(catalog.stats.authors.includes('solanaclawd'));
    assert.equal(catalog.stats.model_downloads_all_time, 11);
    const dumped = JSON.stringify(catalog);
    assert.equal(dumped.includes('hf_live_fixture_token_do_not_leak'), false);
    assert.equal(dumped.includes('Authorization'), false);
    const authors = new URLSearchParams();
    for (const call of stub.calls) {
      if (call.url.includes('author=')) authors.append('seen', new URL(call.url).searchParams.get('author'));
    }
    assert.ok([...new Set([...authors.getAll('seen')])].includes('themusebook'));
  } finally {
    stub.restore();
    restoreEnv('HF_TOKEN', previousToken);
    restoreEnv('HF_AUTHORS', previousAuthors);
  }
});

test('a failed Hub fetch falls back to the dated snapshot with computed stats', async () => {
  resetHubCache();
  const previousToken = process.env.HF_TOKEN;
  delete process.env.HF_TOKEN;
  const stub = installFetch(() => { throw new Error('network down'); });
  try {
    const catalog = await hubCatalog();
    assert.equal(catalog.live, false);
    assert.equal(catalog.degraded, true);
    assert.ok(catalog.stats.models > 0);
    assert.equal(catalog.stats.model_downloads_30d, catalog.models.reduce((sum, model) => sum + (model.downloads || 0), 0));
  } finally {
    stub.restore();
    restoreEnv('HF_TOKEN', previousToken);
  }
});
