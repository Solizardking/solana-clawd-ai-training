import test from 'node:test';
import assert from 'node:assert/strict';
import { SOL, USDC, CHAIN, SIGN_FEATURE, base58Encode, base58Decode, validMint, toAtomic, fromAtomic, compatibleAccount, compatibleWallet, transactionInfo, validateQuote, canApprove, confirmationState } from '../src/trade-core.mjs';

test('amount conversion never rounds token units through Number', () => {
  assert.equal(toAtomic('9007199254.740993',6),'9007199254740993');
  assert.equal(fromAtomic('9007199254740993',6),'9007199254.740993');
  assert.equal(toAtomic('0.000000001',9),'1');
  assert.equal(fromAtomic('1',9),'0.000000001');
  assert.equal(toAtomic('1',0),'1');
  assert.equal(fromAtomic('1000000',6),'1');
  assert.equal(toAtomic('9223372036854775807',0),'9223372036854775807');
  for (const value of ['1e2','-1','0','0.00','1,000','01','1.','NaN','Infinity','9223372036854775808']) assert.throws(() => toAtomic(value,0));
  assert.throws(() => toAtomic('0.0000001',6));
  assert.throws(() => toAtomic('1',-1));
});
test('base58 addresses validate decoded byte length', () => {
  assert.ok(validMint(SOL)); assert.ok(validMint(USDC));
  assert.ok(validMint('11111111111111111111111111111111'));
  assert.equal(validMint('1'.repeat(44)),false);
  assert.equal(validMint('0'.repeat(32)),false);
  assert.equal(validMint(SOL+'1'),false);
  for (const data of [new Uint8Array(32),new Uint8Array([0,0,1,255]),new Uint8Array(64).fill(250)]) assert.deepEqual(base58Decode(base58Encode(data)),data);
});
function fixture() {
  const request = { inputMint:SOL,outputMint:USDC,userPublicKey:SOL,amount:'10000000',displayAmount:'0.01',slippageBps:'auto',revision:1 };
  const order = { executionMode:'sync',inputMint:SOL,outputMint:USDC,inAmount:'10000000',outAmount:'1000000',otherAmountThreshold:'990000',slippageBps:100,priceImpactPct:'0.002',lastValidBlockHeight:100,prioritizationFeeLamports:5000,transaction:'AQ==' };
  return { request,payload:{order,inputDecimals:9,outputDecimals:6,expiresAt:31_000} };
}
test('review rejects mismatched amount, mint, precision, expiry, or execution mode', () => {
  const {request,payload} = fixture();
  const q = validateQuote(payload,request,1000);
  assert.equal(q.expiresAt,31_000);
  for (const field of ['inputMint','outputMint','inAmount','executionMode']) assert.throws(() => validateQuote({...payload,order:{...payload.order,[field]:'wrong'}},request,1000));
  assert.throws(() => validateQuote({...payload,inputDecimals:6},request,1000));
  assert.throws(() => validateQuote({...payload,expiresAt:999},request,1000));
  assert.throws(() => validateQuote({...payload,order:{...payload.order,otherAmountThreshold:'1000001'}},request,1000));
  assert.throws(() => validateQuote(payload,{...request,slippageBps:'50'},1000));
  assert.equal(validateQuote({...payload,expiresAt:90_000},request,1000).expiresAt,31_000);
});
test('sign gate rejects stale quote, changed account/form, and duplicate submission', () => {
  const {request,payload} = fixture(); const quote = validateQuote(payload,request,1000);
  const state = {now:2000,address:SOL,revision:1,busy:false};
  assert.equal(canApprove(quote,state),true);
  assert.equal(canApprove(quote,{...state,now:31_000}),false);
  assert.equal(canApprove(quote,{...state,address:USDC}),false);
  assert.equal(canApprove(quote,{...state,revision:2}),false);
  assert.equal(canApprove(quote,{...state,busy:true}),false);
  assert.equal(canApprove(null,state),false);
});
test('confirmation distinguishes sent, processed, confirmed, failed and expired', () => {
  assert.equal(confirmationState(null,99,100),'pending');
  assert.equal(confirmationState(null,101,100),'expired');
  assert.equal(confirmationState({err:null,confirmationStatus:'processed'},101,100),'pending');
  assert.equal(confirmationState({err:null,confirmationStatus:'confirmed'},101,100),'confirmed');
  assert.equal(confirmationState({err:null,confirmationStatus:'finalized'},99,100),'confirmed');
  assert.equal(confirmationState({err:{InstructionError:[1,'Custom']}},99,100),'failed');
  assert.equal(confirmationState(null,null,100),'pending');
});
test('transaction header binds v0 and legacy transactions to the connected fee payer', () => {
  for (const version of [0,'legacy']) {
    const tx = new Uint8Array(160); tx[0]=1;
    let offset=65; if(version===0) tx[offset++]=128;
    tx[offset++]=1;tx[offset++]=0;tx[offset++]=0;tx[offset++]=1;
    tx.set(base58Decode(SOL),offset);
    assert.deepEqual(transactionInfo(tx),{version,feePayer:SOL});
    tx[0]=2; assert.throws(() => transactionInfo(tx));
  }
  assert.throws(() => transactionInfo(new Uint8Array(3)));
  const tx = new Uint8Array(160);tx[0]=1;tx[65]=129; assert.throws(() => transactionInfo(tx));
});
test('only Wallet Standard mainnet signing accounts are eligible', () => {
  const account = {address:SOL,chains:[CHAIN],features:[SIGN_FEATURE]};
  assert.ok(compatibleAccount(account));
  assert.equal(compatibleAccount({...account,chains:['solana:devnet']}),false);
  assert.equal(compatibleAccount({...account,features:['solana:signTransaction']}),false);
  const wallet = {chains:[CHAIN],features:{'standard:connect':{},'standard:events':{},[SIGN_FEATURE]:{signAndSendTransaction(){}}}};
  assert.ok(compatibleWallet(wallet));
  assert.equal(compatibleWallet({...wallet,chains:['solana:devnet']}),false);
});
