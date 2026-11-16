import json,hashlib
from pathlib import Path
from collections import Counter
from data import read,write,windows,label,ROOT

def account(row):return (row.get('series') or {}).get('account_id') or row['call_id'].split(':')[0]

def main():
 split=json.loads((ROOT/'split.json').read_text());train=read(ROOT/'case_data/train.jsonl');dev_ids=set(split['dev_call_ids']);train_ids=set(split['train_call_ids'])
 assert train_ids.isdisjoint(dev_ids) and train_ids|dev_ids==train.keys()
 assert {account(train[c]) for c in train_ids}.isdisjoint({account(train[c]) for c in dev_ids})
 stats={}
 for name,ids in [('train',split['train_call_ids']),('dev',split['dev_call_ids'])]:
  examples=[]
  for cid in ids:
   call=train[cid];inp={k:v for k,v in call.items() if k!='facts'}
   for window in windows(inp):examples.append({**window,'answer':label(call,window)})
  write(ROOT/f'prepared/{name}.jsonl',examples)
  write(ROOT/f'prepared/{name}_input.jsonl',({k:v for k,v in train[c].items() if k!='facts'} for c in ids))
  if name=='dev':write(ROOT/'prepared/dev_gold.jsonl',(train[c] for c in ids))
  stats[name]={'calls':len(ids),'windows':len(examples),'empty_windows':sum(r['answer']=='[]' for r in examples)}
 val=read(ROOT/'case_data/val.jsonl');assert {account(r) for r in val.values()}.isdisjoint({account(r) for r in train.values()})
 write(ROOT/'prepared/val_input.jsonl',({k:v for k,v in r.items() if k!='facts'} for r in val.values()))
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'case_data').glob('*') if p.is_file()}
 (ROOT/'prepared/data_manifest.json').write_text(json.dumps({'stats':stats,'sha256':manifest},indent=2));print(stats)
if __name__=='__main__':main()
