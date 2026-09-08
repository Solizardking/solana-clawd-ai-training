import pytest
from chart_agent.spark_protocol import parse_tool_calls
from chart_agent.tools import TOOL_DEFS

CALL = '<tool_call>convert_token_amount<arg_key>amount</arg_key><arg_value>18446744073709551615</arg_value><arg_key>decimals</arg_key><arg_value>6</arg_value><arg_key>direction</arg_key><arg_value>raw_to_decimal</arg_value></tool_call>'


def test_exact_amount_and_typed_arguments():
    import json
    call = parse_tool_calls(CALL, TOOL_DEFS)[0]
    args = json.loads(call['function']['arguments'])
    assert args == {'amount':'18446744073709551615','decimals':6,'direction':'raw_to_decimal'}


@pytest.mark.parametrize('value', [CALL[:-12], CALL+ ' stray', CALL.replace('<arg_value>6</arg_value>', '<arg_value>true</arg_value>'), CALL.replace('<arg_key>direction</arg_key><arg_value>raw_to_decimal</arg_value>', ''), CALL.replace('convert_token_amount','invented_tool'), CALL.replace('</tool_call>', '<arg_key>decimals</arg_key><arg_value>9</arg_value></tool_call>')])
def test_invalid_calls_rejected(value):
    with pytest.raises(ValueError):
        parse_tool_calls(value, TOOL_DEFS)


def test_plain_response_is_not_tool():
    assert parse_tool_calls('Chart values are unavailable.', TOOL_DEFS) == []


def test_chart_api_executes_exact_spark_conversion(monkeypatch):
    import asyncio
    import json
    import httpx
    import chart_agent.server as server
    calls = parse_tool_calls(CALL, TOOL_DEFS)
    seen=[]
    def handler(request):
        payload=json.loads(request.content);seen.append(payload)
        message={'role':'assistant','content':None,'tool_calls':calls} if len(seen)==1 else {'role':'assistant','content':'18446744073709.551615'}
        return httpx.Response(200,json={'choices':[{'finish_reason':'tool_calls' if len(seen)==1 else 'stop','message':message}]})
    real=httpx.AsyncClient
    monkeypatch.setattr(server.httpx,'AsyncClient',lambda **kw:real(transport=httpx.MockTransport(handler),**kw))
    monkeypatch.setattr(server,'model_backend','spark')
    answer=asyncio.run(server.analyze_impl(server.AnalyzeRequest(question='Convert the raw amount',use_research=False)))
    assert answer['tools_used'][0]['result']['decimal_amount']=='18446744073709.551615'
    assert seen[1]['messages'][-1]['role']=='tool'
