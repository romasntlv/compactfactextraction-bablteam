import sys,json,argparse,statistics,math
from pathlib import Path
from datetime import datetime
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.codec import read,metrics
from src.extractor import validate_facts

def score(gold_path,pred_path,output):
    gold=read(gold_path);pred=read(pred_path)
    if gold.keys()!=pred.keys():raise ValueError('Missing/extra call ids')
    for cid,row in pred.items():validate_facts(gold[cid],row['facts'])
    metrics.load_catalog(str(ROOT/'evaluation/catalog.json'))
    result=metrics.evaluate(gold,pred)
    result['temporal_correct']=round((result['temporal_recall'] or 0)*result['n_temporal']);result['state_correct']=round((result['state_accuracy'] or 0)*result['n_state'])
    result['empty_facts_rows']=sum(not r['facts'] for r in pred.values());result['schema_valid_calls']=len(pred)
    result['thresholds_pass']={'temporal_mentor':(result['temporal_recall'] or 0)>=.44707,'temporal_readme':(result['temporal_recall'] or 0)>=.4506,'state':(result['state_accuracy'] or 0)>=.5145}
    result['by_attribute']={}
    for attr in ('service_state','tariff','ticket_status','action','agreement'):
        g={c:{**r,'facts':[f for f in r['facts'] if f['attribute']==attr]} for c,r in gold.items()};p={c:{**r,'facts':[f for f in r['facts'] if f['attribute']==attr]} for c,r in pred.items()}
        result['by_attribute'][attr]=metrics.evaluate(g,p)
    errors=[];counts=Counter()
    for cid,g in gold.items():
        pf=pred[cid]['facts'];pairs=metrics.match_facts(g['facts'],pf);matchedg={id(a) for a,b,s in pairs};matchedp={id(b) for a,b,s in pairs}
        recovered=set()
        for t in g['turns']:recovered |= metrics.recover_dates(t['text'],datetime.fromisoformat(g['started_at']).date())
        rd={d.isoformat() for d in recovered}
        for a,b,s in pairs:
            gd={metrics._day(a.get(k)) for k in ('valid_from','valid_to')}-{None}
            if a['attribute'] in metrics.TEMPORAL_ATTRS and rd & gd:
                wrong=[k for k in ('valid_from','valid_to') if metrics._day(a.get(k))!=metrics._day(b.get(k))]
                if wrong:
                    kind='+'.join(wrong);counts[kind]+=1;errors.append({'call_id':cid,'kind':kind,'gold':a,'pred':b})
            if a['source']['turn_idx']!=b['source']['turn_idx']:counts['source_turn']+=1
        for f in g['facts']:
            if id(f) not in matchedg:counts['unmatched_gold']+=1;errors.append({'call_id':cid,'kind':'unmatched_gold','gold':f})
        for f in pf:
            if id(f) not in matchedp:counts['unmatched_prediction']+=1;errors.append({'call_id':cid,'kind':'unmatched_prediction','pred':f})
    result['error_counts']=dict(counts)
    timings=Path(str(pred_path)+'.timings.jsonl')
    if timings.exists():
        rows=[json.loads(s) for s in timings.read_text().splitlines()];assert {r['call_id'] for r in rows}==pred.keys() and len(rows)==len(pred)
        ts=sorted(r['seconds'] for r in rows)
        result['timing']={'n':len(ts),'mean_s':statistics.mean(ts),'p50_s':statistics.median(ts),'p95_s':ts[math.ceil(.95*len(ts))-1],'p99_s':ts[math.ceil(.99*len(ts))-1],'calls_per_hour':3600*len(ts)/sum(ts),'one_device_wall_seconds_per_1000_calls':1000*sum(ts)/len(ts),'partial_calls':sum(r['partial'] for r in rows),'percentile_method':'nearest rank; median averaged','scope':'synchronized extraction incl tokenization/decoding; excludes file IO/load/HTTP; cold call flagged per row'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    Path(str(output)+'.errors.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors))
    print(json.dumps({k:v for k,v in result.items() if k!='by_attribute'},ensure_ascii=False,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('gold');p.add_argument('pred');p.add_argument('output');a=p.parse_args();score(a.gold,a.pred,a.output)
