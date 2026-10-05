import json,os,sqlite3,threading,time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel,Field
from .core import route,models
from .benchmark import profile_rows
LOCK=threading.Lock()
def db():
    c=sqlite3.connect(os.getenv('ROUTEGUARD_DB','routeguard.db'),timeout=10)
    c.execute('CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, created REAL, task TEXT, status TEXT, cost REAL, fallbacks INTEGER, payload TEXT)');c.commit();return c
@asynccontextmanager
async def lifespan(app):
    c=db();c.close();yield
app=FastAPI(title='RouteGuard AI',version='2.0.0',lifespan=lifespan)
class Request(BaseModel):
    prompt:str=Field(min_length=1,max_length=20000)
    context:str=Field(default='',max_length=50000)
    quality_threshold:float=Field(default=.9,ge=.5,le=1)
    expected_answer:str|None=None
    expected_json:bool=False
    required_terms:list[str]=Field(default_factory=list,max_length=30)
    max_attempts:int=Field(default=3,ge=1,le=7)
    use_judge:bool=True
@app.get('/health')
def health():return {'ok':True,'demo_mode':os.getenv('DEMO_MODE','true').lower()=='true','judge_configured':bool(os.getenv('JUDGE_MODEL')),'models':[{'model':m.model,'input_per_m':m.input_per_m,'output_per_m':m.output_per_m} for m in models()]}
@app.post('/route')
async def route_endpoint(req:Request):
    result=await route(**req.model_dump())
    with LOCK:
        c=db()
        try:c.execute('INSERT INTO runs(created,task,status,cost,fallbacks,payload) VALUES(?,?,?,?,?,?)',(time.time(),result['task'],result['status'],result['known_cost_usd'],result['fallback_count'],json.dumps(result)));c.commit()
        finally:c.close()
    return result
@app.get('/metrics')
def metrics():
    with LOCK:
        c=db()
        try:
            n,cost,passed,fall=c.execute("SELECT COUNT(*),COALESCE(SUM(cost),0),COALESCE(SUM(CASE WHEN status='passed' THEN 1 ELSE 0 END),0),COALESCE(SUM(fallbacks),0) FROM runs").fetchone()
            rows=c.execute('SELECT id,created,task,status,cost,fallbacks FROM runs ORDER BY id DESC LIMIT 100').fetchall()
        finally:c.close()
    return {'requests':n,'total_cost_usd':cost,'verified_pass_rate':passed/n if n else 0,'fallbacks':fall,'recent':[dict(zip(['id','created','task','status','cost','fallbacks'],r)) for r in rows]}

@app.get("/benchmarks/profiles")
def benchmark_profiles():
    return {"profiles":profile_rows(),"minimum_samples":int(os.getenv("MIN_BENCHMARK_SAMPLES","5")),"note":"Wilson lower bound is conservative; not a per-prompt quality guarantee."}
