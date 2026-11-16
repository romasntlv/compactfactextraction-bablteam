import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.codec import read,write

def subset(source,ids_file,output):
 ids=read(ids_file);rows=read(source);write(output,(rows[c] for c in ids))
 timing=Path(str(source)+'.timings.jsonl')
 if timing.exists():
  ts=read(timing);write(str(output)+'.timings.jsonl',(ts[c] for c in ids))
 for suffix in ('.segments.jsonl',):
  p=Path(str(source)+suffix)
  if p.exists():shutil.copy2(p,str(output)+suffix)
 # Segment peak memory belongs to full source run, not this subset; report accordingly.
if __name__=='__main__':subset(*sys.argv[1:])
