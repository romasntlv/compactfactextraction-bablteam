"""Check new release codec on archived model strings without running a model."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.codec import read,parse,dedup
from src.extractor import validate_facts

def main():
 report={}
 for split,ref,raw in [('dev','reports/dev_checkpoint-2142.jsonl','reports/dev_checkpoint-2142.jsonl.raw.jsonl'),('val','data/val_reference.jsonl','reports/val_facts.jsonl.raw.jsonl')]:
  calls=read(ROOT/f'data/{split}_input.jsonl');reference=read(ROOT/ref);facts={cid:[] for cid in calls};stats=Counter()
  for s in (ROOT/raw).read_text().splitlines():
   r=json.loads(s);parsed,err=parse(r['text'],calls[r['call_id']],r['target']);stats.update(err)
   for f in parsed:
    if f['object']['type']!='other':facts[r['call_id']].append(f)
    else:stats['invalid_rows']+=1
  for cid in calls:
   facts[cid]=dedup(facts[cid]);validate_facts(calls[cid],facts[cid]);assert facts[cid]==reference[cid]['facts'],cid
  report[split]={'exact_equal_calls':len(calls),'codec_errors':dict(stats)}
 (ROOT/'reports/codec_regression.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
