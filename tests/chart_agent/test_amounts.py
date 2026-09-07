import pytest
from chart_agent.tools import convert_token_amount

@pytest.mark.parametrize('raw,decimals,expected',[
    ('18446744073709551615',6,'18446744073709.551615'),
    ('123456789012345678',8,'1234567890.12345678'),
    ('9007199254740993',9,'9007199.254740993'),
    ('1',9,'0.000000001'),('0',0,'0')])
def test_exact_amounts_roundtrip(raw,decimals,expected):
    assert convert_token_amount(raw,decimals,'raw_to_decimal')['decimal_amount']==expected
    assert convert_token_amount(expected,decimals,'decimal_to_raw')['raw_amount']==raw

@pytest.mark.parametrize('amount,decimals,direction',[
    ('18446744073709551616',6,'raw_to_decimal'),('1e9',9,'raw_to_decimal'),
    ('-1',9,'raw_to_decimal'),('0.0000000001',9,'decimal_to_raw'),
    ('NaN',9,'decimal_to_raw'),('1',True,'raw_to_decimal'),
    ('1',256,'raw_to_decimal'),(9007199254740993,9,'raw_to_decimal')])
def test_reject_unsafe_or_inexact_inputs(amount,decimals,direction):
    with pytest.raises(ValueError):convert_token_amount(amount,decimals,direction)

def test_trailing_zero_precision_is_exact():
    assert convert_token_amount('1.0000000000',9,'decimal_to_raw')['raw_amount']=='1000000000'
