"""Live HTTP demo: no replay, no gold, standard library only."""
import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',default=str(Path(__file__).resolve().parents[1]/'data/example_request.json'));p.add_argument('--url',default='http://127.0.0.1:8000/extract');a=p.parse_args()
call=json.loads(Path(a.input).read_text());assert 'facts' not in call
print(json.dumps(call,ensure_ascii=False,indent=2),flush=True)
req=urllib.request.Request(a.url,data=json.dumps(call).encode(),headers={'Content-Type':'application/json'})
t=time.perf_counter()
with urllib.request.urlopen(req,timeout=600) as response:
 result=json.load(response);partial=response.headers.get('X-CFE-Partial');model_seconds=response.headers.get('X-CFE-Model-Seconds')
print(json.dumps(result,ensure_ascii=False,indent=2));print(f'HTTP seconds: {time.perf_counter()-t:.3f}; model seconds: {model_seconds}; partial: {partial}')
