"""Separate official scoring and data-integrity checks."""
import json,argparse
from pathlib import Path
from data import read,metrics

def evaluate(gold_path,pred_path):
 gold,pred=read(gold_path),read(pred_path);assert gold.keys()==pred.keys(),'Missing/extra call_id'
 for cid,row in pred.items():
  assert isinstance(row['facts'],list);turns={t['idx']:t for t in gold[cid]['turns']}
  assert len(turns)==len(gold[cid]['turns'])
  for f in row['facts']:
   assert f['source']['turn_idx'] in turns
   assert f['source']['quote']==turns[f['source']['turn_idx']]['text']
   assert f['valid_to'] is None or f['valid_from'][:10]<=f['valid_to'][:10]
 metrics.load_catalog(str(Path(__file__).resolve().parent/'case_data/catalog.json'))
 result=metrics.evaluate(gold,pred);result['temporal_ok']=round((result['temporal_recall'] or 0)*result['n_temporal']);result['state_ok']=round((result['state_accuracy'] or 0)*result['n_state']);result['empty_facts_rows']=sum(not p['facts'] for p in pred.values());result['temporal_pass']=(result['temporal_recall'] or 0)>=.44707;result['state_pass']=(result['state_accuracy'] or 0)>=.5145
 return result

def key(r):
 # Optimize the weakest relative target, then temporal, then state, then fact F1.
 t=r['temporal_recall'] or 0;s=r['state_accuracy'] or 0;p=r['fact_precision'];rec=r['fact_recall']
 return (min(t/.44707,s/.5145),t,s,2*p*rec/max(p+rec,1e-9))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('gold');p.add_argument('pred');p.add_argument('--output');args=p.parse_args();r=evaluate(args.gold,args.pred);print(json.dumps(r,indent=2))
 if args.output:Path(args.output).write_text(json.dumps(r,indent=2))
