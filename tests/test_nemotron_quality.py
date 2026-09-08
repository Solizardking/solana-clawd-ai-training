import unittest
from scripts.evaluate_nemotron_quality import score_response


def response(content, finish='stop', calls=None):
    return {'choices':[{'finish_reason':finish,'message':{'content':content,'tool_calls':calls}}]}


class QualityScoringTests(unittest.TestCase):
    def test_exact_json(self):
        self.assertTrue(score_response(response('{"decimals":null}'), {'decimals':None}))
        self.assertTrue(score_response(response('```json\n{"decimals":null}\n```'), {'decimals':None}))

    def test_reject_numeric_boolean_substitution(self):
        self.assertFalse(score_response(response('{"fresh":0}'), {'fresh':False}))

    def test_reject_truncation_and_tool_response(self):
        self.assertFalse(score_response(response('{"fresh":false}', 'length'), {'fresh':False}))
        self.assertFalse(score_response(response('{"fresh":false}', calls=[{}]), {'fresh':False}))

    def test_reject_wrong_or_unparseable_answer(self):
        for text in ['{"decimals":8}', 'I think null', '{"other":null}', '{"decimals":null,"guess":8}']:
            self.assertFalse(score_response(response(text), {'decimals':None}))

if __name__=='__main__':unittest.main()
