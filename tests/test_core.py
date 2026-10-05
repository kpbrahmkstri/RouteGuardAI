import asyncio
from routeguard.core import analyze,deterministic,rank_models,models,route,price,Model,needs_llm_judge

def test_seven_models():
    assert len(models()) == 7
    assert len({m.model for m in models()}) == 7

def test_factual_qa_classification():
    p=analyze('What is the capital of Oman?')
    assert p['task']=='factual_qa' and p['complexity']=='low'


def test_factual_reference_skips_judge():
    assert not needs_llm_judge(
        'factual_qa',
        expected_answer='Muscat'
    )


def test_explanation_requires_judge():
    assert needs_llm_judge(
        'knowledge_explanation'
    )


def test_structured_output_can_skip_judge():
    assert not needs_llm_judge(
        'structured_output',
        expected_json=True
    )

def test_cold_start_uses_cost_not_fake_quality(monkeypatch,tmp_path):
    monkeypatch.setenv('ROUTEGUARD_DB',str(tmp_path/'empty.db'))
    p=analyze('What is the capital of Oman?')
    ranked,rows,met,mode=rank_models(p,models(),.90,'What is the capital of Oman?','')
    assert not met and mode=='exploration_cold_start'
    assert ranked[0]['model']=='gpt-4.1-nano'
    assert all(x['quality_lower_bound'] is None for x in rows)

def test_knowledge_explanation_classification():
    p = analyze('Explain photosynthesis')
    assert p['task'] == 'knowledge_explanation'
    assert p['complexity'] == 'low'


def test_general_fallback():
    p = analyze('Tell me something interesting')
    assert p['task'] == 'general'

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

def test_objective_answer_skips_judge_and_passes(monkeypatch,tmp_path):
    monkeypatch.setenv('DEMO_MODE','false')
    monkeypatch.setenv('ROUTEGUARD_DB',str(tmp_path/'empty.db'))
    async def generate_answer(*args,**kwargs):
        return 'Muscat',10,2,0.01

    async def unexpected_judge(*args,**kwargs):
        raise AssertionError('Judge should be skipped for an objectively checked answer')

    monkeypatch.setattr('routeguard.core.generate',generate_answer)
    monkeypatch.setattr('routeguard.core.judge',unexpected_judge)
    result=asyncio.run(route(
        'What is the capital of Oman?',
        expected_answer='Muscat',
        use_judge=True,
        max_attempts=1,
    ))
    attempt=result['attempts'][0]
    assert result['status']=='passed'
    assert attempt['passed'] is True
    assert attempt['judge']['status']=='not_required'
    assert attempt['judge']['cost_usd']==0
    assert attempt['judge']['input_tokens']==0
