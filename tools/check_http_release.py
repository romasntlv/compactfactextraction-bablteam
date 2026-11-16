import sys,json,time,argparse
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.extractor import validate_facts
from tools.benchmark_http import main as benchmark

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18080');a=p.parse_args()
 with httpx.Client(timeout=1800) as client:
  for i in range(60):
   try:
    if client.get(a.url+'/health').status_code==200:break
   except httpx.ConnectError:pass
   time.sleep(.5)
  else:raise RuntimeError('service not ready')
  empty={'call_id':'empty','started_at':'2026-05-20T12:00:00','turns':[]}
  response=client.post(a.url+'/extract',json=empty);assert response.status_code==200 and response.json()=={'call_id':'empty','facts':[]}
  assert client.post(a.url+'/extract',json={**empty,'facts':[]}).status_code==422
  assert client.post(a.url+'/extract',json={**empty,'started_at':'bad'}).status_code==422
 sample=ROOT/'data/http3_input.jsonl';sample.write_text('\n'.join((ROOT/'data/curve24_input.jsonl').read_text().splitlines()[:3])+'\n')
 target=ROOT/'benchmarks/http_docker_cpu.jsonl'
 benchmark(['--url',a.url,'--input',str(sample),'--output',str(target)])
 calls={c['call_id']:c for c in map(json.loads,sample.read_text().splitlines())}
 for row in map(json.loads,target.read_text().splitlines()):validate_facts(calls[row['call_id']],row['response']['facts'])
 (ROOT/'reports/http_docker_checks.json').write_text(json.dumps({'health':True,'empty_turns_empty_facts':True,'reject_gold_422':True,'reject_invalid_date_422':True,'real_model_calls':3,'source_schema_valid':True,'scope':'Linux ARM64 Docker CPU on Mac; not x86_64/CUDA'},indent=2))
 print('Real Docker HTTP contract and source validation PASS')
if __name__=='__main__':main()
