import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path
spec=importlib.util.spec_from_file_location('gateway',Path(__file__).with_name('gateway.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
SOL='So11111111111111111111111111111111111111112'
USDC='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
class GatewayTests(unittest.TestCase):
    def body(self):return {'inputMint':SOL,'outputMint':USDC,'userPublicKey':SOL,'amount':'1000000'}
    def test_amounts_do_not_round(self):
        b=self.body();b['amount']='9007199254740993'
        self.assertEqual(g.validate_order(b)['amount'],'9007199254740993')
        for x in [1.5,0,'0','-1','1.1','1e9','9223372036854775808']:
            b['amount']=x
            with self.assertRaises(ValueError):g.validate_order(b)
    def test_wallet_mint_and_pair_validation(self):
        for key in ('inputMint','outputMint','userPublicKey'):
            b=self.body();b[key]='not-a-mint'
            with self.assertRaises(ValueError):g.validate_order(b)
        b=self.body();b['outputMint']=SOL
        with self.assertRaises(ValueError):g.validate_order(b)
        self.assertFalse(g.valid_address('z'*44))
    def test_trade_request_cannot_add_recipient_or_disable_safety(self):
        b=self.body();b.update(destinationWallet=USDC,skipSimulation=True,platformFeeBps=10000,prioritizationFeeMaxLamports=999999999)
        p=g.validate_order(b)
        self.assertNotIn('destinationWallet',p)
        self.assertNotIn('skipSimulation',p)
        self.assertNotIn('platformFeeBps',p)
        self.assertEqual(p['prioritizationFeeMaxLamports'],'5000000')
    def test_response_must_match_review_and_minimum(self):
        p=g.validate_order(self.body())
        o={**p,'inAmount':p['amount'],'outAmount':'100','otherAmountThreshold':'90','executionMode':'sync','transaction':'test','lastValidBlockHeight':1,'slippageBps':50,'prioritizationFeeLamports':1000}
        g.validate_response(o,p)
        for key,value in [('inputMint',USDC),('inAmount','999'),('executionMode','async'),('otherAmountThreshold','0'),('prioritizationFeeLamports',5000001)]:
            with self.assertRaises(ValueError):g.validate_response({**o,key:value},p)
    def test_token_decimals_are_authoritative_and_cached(self):
        g.TOKENS={}
        mint='8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump'
        with patch.object(g,'upstream_json',return_value={'result':{'value':{'decimals':6}}}) as api:
            self.assertEqual(g.decimals(SOL),9);self.assertEqual(g.decimals(USDC),6)
            self.assertEqual(g.decimals(mint),6);self.assertEqual(g.decimals(mint),6)
            api.assert_called_once()
if __name__=='__main__':unittest.main()
