from scripts.review_direct_bar_questions import check


def observation():
    return {'image':'images/syn_bar_fixture.png','size':[500,500], 'chart_elements':[],
      'ocr':{'available':True,'words':[
        {'text':'Retail','confidence':.95,'bbox':[100,450,140,470]},
        {'text':'$4900','confidence':.95,'bbox':[100,250,140,270]}]}}


def messages():return [{'role':'user','content':'How much revenue did Retail generate?'},{'role':'assistant','content':'$4900'}]


def test_direct_value_is_grounded():
    assert check(messages(),observation())['value']=='$4900'


def test_wrong_category_alignment_rejected():
    obs=observation();obs['ocr']['words'][1]['bbox']=[300,250,340,270]
    assert check(messages(),obs) is None


def test_ambiguous_value_rejected():
    obs=observation();obs['ocr']['words'].append({'text':'$9000','confidence':.99,'bbox':[100,200,140,220]})
    assert check(messages(),obs) is None


def test_gold_target_does_not_override_observations():
    msg=messages();msg[1]['content']='$490'
    assert check(msg,observation()) is None


def test_comparison_question_not_treated_as_lookup():
    msg=messages();msg[0]['content']='How much more revenue did Retail generate than Direct?'
    assert check(msg,observation()) is None
