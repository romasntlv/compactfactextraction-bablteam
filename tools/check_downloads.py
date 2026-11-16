"""Verify downloaded release files against SHA256SUMS (run from repo root)."""
import hashlib,sys
from pathlib import Path
expected={}
for line in Path('SHA256SUMS').read_text().splitlines():
 digest,name=line.split(maxsplit=1);expected[name.lstrip('*')]=digest
for name in sys.argv[1:]:
 p=Path(name);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=expected[p.name]:raise SystemExit('CHECKSUM MISMATCH: '+name)
 print('OK:',name)
if len(sys.argv)==1:raise SystemExit('Specify one or more downloaded filenames')
