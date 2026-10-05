"""Run labeled JSONL prompts through the API and export transparent benchmark results."""
import argparse,csv,json,statistics
import httpx

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='data/sample.jsonl');ap.add_argument('--url',default='http://localhost:8000');ap.add_argument('--output',default='benchmark_results.csv');args=ap.parse_args()
    rows=[]
    with httpx.Client(timeout=160) as client,open(args.dataset) as f:
        for line in f:
            if not line.strip():continue
            case=json.loads(line);result=client.post(args.url+'/route',json=case).raise_for_status().json()
            rows.append({'task':result['task'],'status':result['status'],'cost_usd':result['total_cost_usd'],'fallbacks':result['fallback_count'],'attempts':len(result['attempts']),'demo_mode':result['demo_mode']})
    if not rows:raise SystemExit('Empty dataset')
    with open(args.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps({'cases':len(rows),'proxy_pass_rate':sum(x['status']=='passed' for x in rows)/len(rows),'total_cost_usd':sum(x['cost_usd'] for x in rows),'mean_fallbacks':statistics.mean(x['fallbacks'] for x in rows),'demo_mode':rows[0]['demo_mode'],'results_file':args.output},indent=2))
if __name__=='__main__':main()
