import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_QUESTIONS, rejectSecretPayload, resolveSitePage, normalizeAnswers, isMint, CLAWD_MINT, clefConfigured } from '../server/clef.mjs';

test('default research questions cover noul, choice and score', () => {
  assert.equal(DEFAULT_QUESTIONS.safety_check.type, 'noul');
  assert.equal(DEFAULT_QUESTIONS.liquidity_ok.type, 'noul');
  assert.equal(DEFAULT_QUESTIONS.momentum.type, 'score');
  assert.equal(DEFAULT_QUESTIONS.distribution.type, 'score');
  assert.equal(DEFAULT_QUESTIONS.verdict.type, 'choice');
  assert.ok(DEFAULT_QUESTIONS.verdict.criteria.watch);
  assert.ok(isMint(CLAWD_MINT));
});

test('rejects signing material as a credential payload', () => {
  const payload = "-----BEGIN " + "PRIVATE KEY-----\nsynthetic-test-data";
  assert.throws(() => rejectSecretPayload(payload), /credential_payload/);
  assert.throws(() => rejectSecretPayload({ state: payload }), /credential_payload/);
});

test('host routing serves the playground and index without replacing the hub', () => {
  assert.equal(resolveSitePage('clef.onchainai.fund', '/'), '/clef/index.html');
  assert.equal(resolveSitePage('index.onchainai.fund', '/'), '/decision-index/index.html');
  assert.equal(resolveSitePage('onchainai.fund', '/clef'), '/clef/index.html');
  assert.equal(resolveSitePage('onchainai.fund', '/decision-index'), '/decision-index/index.html');
  assert.equal(resolveSitePage('onchainai.fund', '/'), null);
  assert.equal(resolveSitePage('onchainai.fund', '/index.html'), null);
});

test('normalizes choice answers that use answer instead of choice', () => {
  const answers = normalizeAnswers({ verdict: { answer: 'watch', probs: { watch: 0.6, avoid: 0.4 } } }, DEFAULT_QUESTIONS);
  assert.equal(answers.verdict.type, 'choice');
  assert.equal(answers.verdict.choice, 'watch');
});

test('clefConfigured is false without secrets', () => {
  const prevA = process.env.CF_ACCOUNT_ID;
  const prevT = process.env.CF_API_TOKEN;
  delete process.env.CF_ACCOUNT_ID;
  delete process.env.CF_API_TOKEN;
  try { assert.equal(clefConfigured(), false); }
  finally {
    if (prevA === undefined) delete process.env.CF_ACCOUNT_ID; else process.env.CF_ACCOUNT_ID = prevA;
    if (prevT === undefined) delete process.env.CF_API_TOKEN; else process.env.CF_API_TOKEN = prevT;
  }
});
