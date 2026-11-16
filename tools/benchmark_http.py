"""Run against an already started service; includes HTTP JSON serialization/transit."""
import sys,json,argparse,time,statistics,math
from pathlib import Path
import httpx

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
 rows=[];calls=[json.loads(s) for s in Path(a.input).read_text().splitlines()]
 with httpx.Client(timeout=1800) as c,Path(a.output).open('w') as stream:
  c.get(a.url+'/health').raise_for_status()
  for i,call in enumerate(calls):
   t=time.perf_counter();r=c.post(a.url+'/extract',json=call);elapsed=time.perf_counter()-t;r.raise_for_status()
   row={'call_id':call['call_id'],'seconds':elapsed,'model_seconds':float(r.headers['X-CFE-Model-Seconds']),'partial':r.headers['X-CFE-Partial'],'response':r.json()};rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();print(f'HTTP {i+1}/{len(calls)}: {elapsed:.3f}s',flush=True)
 ts=sorted(r['seconds'] for r in rows);report={'calls':len(ts),'mean_s':statistics.mean(ts),'p50_s':statistics.median(ts),'p95_s':ts[math.ceil(.95*len(ts))-1],'calls_per_hour':3600*len(ts)/sum(ts),'scope':'sequential real HTTP, no warmup; first request included; model load before server readiness excluded'}
 Path(a.output+'.summary.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
