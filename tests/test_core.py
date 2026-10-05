import asyncio
from routeguard.core import analyze,deterministic,rank_models,models,route,price,Model

def test_seven_models():
    assert len(models()) == 7
    assert len({m.model for m in models()}) == 7

def test_factual_qa_classification():
    p=analyze('What is the capital of Oman?')
    assert p['task']=='factual_qa' and p['complexity']=='low'

def test_cold_start_uses_cost_not_fake_quality(monkeypatch,tmp_path):
    monkeypatch.setenv('ROUTEGUARD_DB',str(tmp_path/'empty.db'))
    p=analyze('What is the capital of Oman?')
    ranked,rows,met,mode=rank_models(p,models(),.90,'What is the capital of Oman?','')
    assert not met and mode=='exploration_cold_start'
    assert ranked[0]['model']=='gpt-4.1-nano'
    assert all(x['quality_lower_bound'] is None for x in rows)

def test_known_pricing():
    assert price(Model('x','x',1,2,1000,{}),1_000_000,1_000_000)==3
    assert price(models()[0],1_000_000,1_000_000)==0.50

def test_json():
    assert deterministic('{"a":1}',expected_json=True)['valid_json']['passed']

def test_demo(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_MODE','true')
    monkeypatch.setenv('ROUTEGUARD_DB',str(tmp_path/'empty.db'))
    r=asyncio.run(route('What is the capital of Oman?'))
    assert r['status']=='demo_unverified' and len(r['candidates'])==7
    assert r['selected_model']=='gpt-4.1-nano'
    assert r['total_cost_usd'] is not None and r['total_tokens']>0
