"""Versioned offline model benchmarks; only real, evaluated calls populate profiles."""
import asyncio, json, math, os, sqlite3, statistics, time
from pathlib import Path
from .core import Model, models, analyze, generate, deterministic, judge, price

DATASET_VERSION = 'v2'

def connection():
    db=sqlite3.connect(os.getenv('ROUTEGUARD_DB','routeguard.db'),timeout=30)
    db.execute('''CREATE TABLE IF NOT EXISTS benchmark_results (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, dataset_version TEXT NOT NULL,
        case_id TEXT NOT NULL, model TEXT NOT NULL, task TEXT NOT NULL,
        passed INTEGER NOT NULL, generation_cost REAL, judge_cost REAL,
        input_tokens INTEGER, output_tokens INTEGER, latency_s REAL,
        details TEXT, created REAL NOT NULL)''')
    db.commit();return db

def wilson_lower_bound(successes,n,z=1.96):
    if n<=0:return 0.0
    p=successes/n; den=1+z*z/n
    return max(0.,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/den)

def load_profiles():
    db=connection()
    try:
        rows=db.execute('''SELECT model,task,COUNT(*),SUM(passed),AVG(generation_cost),AVG(COALESCE(generation_cost,0)+COALESCE(judge_cost,0)),AVG(latency_s)
          FROM benchmark_results WHERE dataset_version=? GROUP BY model,task''',(DATASET_VERSION,)).fetchall()
    finally:db.close()
    return {(m,t):dict(n=n,passes=passes,mean_generation_cost_usd=cost,mean_total_cost_usd=total_cost,mean_latency_s=latency) for m,t,n,passes,cost,total_cost,latency in rows}

def dataset(path):
    cases=[];ids=set()
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        c=json.loads(line);cid=c['id']
        if cid in ids:raise ValueError('Duplicate benchmark case '+cid)
        ids.add(cid)
        if not c.get('prompt') or not any(c.get(x) for x in ('expected_answer','required_terms','expected_json','context')):
            raise ValueError('Benchmark requires prompt and independently checkable criteria: '+cid)
        cases.append(c)
    if not cases:raise ValueError('Empty benchmark dataset')
    return cases

async def run_benchmark(path='data/benchmark.jsonl',model_ids=None,run_id=None,threshold=.85,use_judge=True):
    if os.getenv('DEMO_MODE','true').lower()=='true':raise ValueError('Benchmarks require DEMO_MODE=false and actual provider calls')
    cases=dataset(path);available={m.model:m for m in models()}
    selected=model_ids or list(available)
    if set(selected)-set(available):raise ValueError('Unknown models: '+str(set(selected)-set(available)))
    run_id=run_id or str(int(time.time()))
    db=connection();summary=[]
    try:
        for case in cases:
            task=case.get('task') or analyze(case['prompt'],case.get('context',''),case.get('expected_json',False))['task']
            for model_id in selected:
                m=available[model_id];t0=time.monotonic()
                try:
                    answer,inp,out,latency=await generate(m,case['prompt'],case.get('context',''),case.get('expected_json',False))
                    checks=deterministic(answer,case.get('context',''),case.get('expected_answer'),case.get('expected_json',False),case.get('required_terms'))
                    evaluation=await judge(case['prompt'],answer,case.get('context',''),case.get('expected_answer')) if use_judge else {'status':'disabled','scores':{},'cost_usd':0,'input_tokens':0,'output_tokens':0}
                    mandatory=all(x['passed'] for x in checks.values())
                    scores={k:v for k,v in evaluation.get('scores',{}).items() if k!='truthfulness' and v is not None}
                    # An evaluator failure never counts as a pass. If disabled, only objective criteria count.
                    passed=mandatory and ((evaluation['status']=='evaluated' and bool(scores) and all(v>=threshold for v in scores.values())) if use_judge else True)
                    gen_cost=price(m,inp,out);judge_cost=evaluation.get('cost_usd')
                    details={'checks':checks,'judge':evaluation,'answer':answer}
                except Exception as exc:
                    passed=False;gen_cost=None;judge_cost=None;inp=out=0;latency=time.monotonic()-t0
                    details={'error':str(exc)[:300]}
                db.execute('''INSERT INTO benchmark_results(run_id,dataset_version,case_id,model,task,passed,generation_cost,judge_cost,input_tokens,output_tokens,latency_s,details,created)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(run_id,DATASET_VERSION,case['id'],model_id,task,int(passed),gen_cost,judge_cost,inp,out,latency,json.dumps(details),time.time()))
                db.commit()
                summary.append({'case_id':case['id'],'model':model_id,'task':task,'passed':passed,'generation_cost_usd':gen_cost,'judge_cost_usd':judge_cost,'latency_s':round(latency,3),'error':details.get('error')})
    finally:db.close()
    return {'run_id':run_id,'dataset_version':DATASET_VERSION,'results':summary,'profiles':profile_rows()}

def profile_rows():
    return [{'model':m,'task':t,'samples':v['n'],'pass_rate':round(v['passes']/v['n'],3),'wilson_lower_bound':round(wilson_lower_bound(v['passes'],v['n']),3),'mean_generation_cost_usd':v['mean_generation_cost_usd'],'mean_total_cost_usd':v['mean_total_cost_usd'],'cost_per_success_usd':(v['mean_total_cost_usd']/(v['passes']/v['n']) if v['mean_total_cost_usd'] is not None and v['passes'] else None),'mean_latency_s':v['mean_latency_s']} for (m,t),v in sorted(load_profiles().items())]

if __name__=='__main__':
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='data/benchmark.jsonl');parser.add_argument('--models',nargs='*');parser.add_argument('--threshold',type=float,default=.85);parser.add_argument('--no-judge',action='store_true')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(run_benchmark(args.dataset,args.models,threshold=args.threshold,use_judge=not args.no_judge)),indent=2))
