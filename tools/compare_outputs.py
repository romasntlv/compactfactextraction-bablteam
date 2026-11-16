import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.codec import read

def compare(reference,prediction,output):
 a,b=read(reference),read(prediction)
 assert a.keys()==b.keys(),'call ids differ'
 changed=[c for c in a if a[c]['facts']!=b[c]['facts']]
 report={'calls':len(a),'exact_equal_calls':len(a)-len(changed),'changed_calls':changed,'note':'Cross-device/dtype equality is diagnostic; score both outputs with official metrics.'}
 Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2));print({k:v for k,v in report.items() if k!='changed_calls'});return report
if __name__=='__main__':compare(*sys.argv[1:])
