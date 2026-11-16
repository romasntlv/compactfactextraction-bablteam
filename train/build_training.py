"""Rebuild the exact augmented training windows and verify the archived hash."""
import hashlib,json
from pathlib import Path
from prepare import main as prepare
from data import read,windows,label,write,ROOT

def main():
    prepare();rows=[]
    for call in read(ROOT/'synthetic_train.jsonl').values():
        inp={k:v for k,v in call.items() if k not in ('facts','synthetic_metadata')}
        rows.extend({**w,'answer':label(call,w)} for w in windows(inp))
    out=ROOT/'prepared/train_augmented.jsonl'
    out.write_text((ROOT/'prepared/train.jsonl').read_text()+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    actual=hashlib.sha256(out.read_bytes()).hexdigest();expected=json.loads((ROOT.parent/'models/training_manifest.json').read_text())['train_sha256']
    assert actual==expected,(actual,expected)
    print('Training windows SHA256 reproduced:',actual)
if __name__=='__main__':main()
