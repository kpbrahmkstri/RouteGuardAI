from routeguard.benchmark import wilson_lower_bound,dataset
from routeguard.core import rank_models,analyze,Model

def test_wilson_conservative():
    assert wilson_lower_bound(5,5)<.6
    assert wilson_lower_bound(100,100)>.95
    assert wilson_lower_bound(0,0)==0

def test_dataset():
    cases = dataset('data/benchmark.jsonl')

    assert len(cases) >= 21

    assert sum(
        c.get('task') == 'factual_qa'
        for c in cases
    ) >= 8

    assert sum(
        c.get('task') == 'knowledge_explanation'
        for c in cases
    ) >= 5

def test_cold_start_is_explicit_exploration(monkeypatch,tmp_path):
    monkeypatch.setenv('ROUTEGUARD_DB',str(tmp_path/'empty.db'))
    cheap=Model('cheap','cheap',.1,.4,128000,{})
    expensive=Model('expensive','expensive',5.,20.,128000,{})
    ordered,rows,met,mode=rank_models(analyze('What is the capital of Oman?'),[expensive,cheap],.9,'What is the capital of Oman?','')
    assert not met and mode=='exploration_cold_start'
    assert ordered[0]['model']=='cheap'
    assert all(r['evidence']=='insufficient_benchmark_evidence' for r in rows)
