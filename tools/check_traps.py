import json,sys
from pathlib import Path
def check(pred,output):
 rows=[json.loads(s) for s in Path(pred).read_text().splitlines()]
 result={'calls':len(rows),'empty_correct':sum(not r['facts'] for r in rows),'failures':[r for r in rows if r['facts']],'scope':'manual all-empty probes only, not general precision/recall'}
 assert len(rows)==5
 Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result)
if __name__=='__main__':check(*sys.argv[1:])
