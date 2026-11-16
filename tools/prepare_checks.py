"""Predeclared, gold-independent resource subset and semantics-preserving noise pilot."""
import sys,json,hashlib,copy,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.codec import read,write

def main():
    split=json.loads((ROOT/'train/split.json').read_text());train=read(ROOT/'train/case_data/train.jsonl');val=read(ROOT/'train/case_data/val.jsonl')
    dev=[train[c] for c in split['dev_call_ids']]
    def inp(rows):return [{k:v for k,v in r.items() if k!='facts'} for r in rows]
    write(ROOT/'data/dev_input.jsonl',inp(dev));write(ROOT/'data/dev_gold.jsonl',dev)
    write(ROOT/'data/val_input.jsonl',inp(val.values()));write(ROOT/'data/val_gold.jsonl',val.values())
    # Stratify by transcript character length, equally spaced across sorted corpus.
    ordered=sorted(dev,key=lambda c:(sum(len(t['text']) for t in c['turns']),c['call_id']))
    def subset(n):return [ordered[min(len(ordered)-1,int((i+.5)*len(ordered)/n))] for i in range(n)]
    small=subset(24);stress=subset(96)
    write(ROOT/'data/curve24_input.jsonl',inp(small));write(ROOT/'data/curve24_gold.jsonl',small)
    write(ROOT/'data/stress96_input.jsonl',inp(stress));write(ROOT/'data/stress96_gold.jsonl',stress)
    edits=[]
    # Do not remove negation, dates, numbers, or content words. Stress test is limited
    # to phonetic spellings of known nouns and filler/word repetition, not arbitrary ASR.
    replacements={'интернет':'интернэт','роутер':'роутир','пожалуйста':'пажалуста','спасибо':'спасиба','подскажите':'падскажите','телевизор':'теливизор','здравствуйте':'здраствуйте'}
    for level,rate in [('mild',.3),('strong',.7)]:
        changed=copy.deepcopy(stress);rng=random.Random(20261115)
        for call in changed:
            for t in call['turns']:
                old=t['text']
                def replace(m):
                    word=m.group();r=replacements.get(word.lower())
                    return r if r and rng.random()<rate else word
                t['text']=re.sub(r'[А-Яа-яЁё]+',replace,old)
                if rng.random()<rate:t['text']='Э-э, '+t['text']
                if t['text']!=old:edits.append({'level':level,'call_id':call['call_id'],'turn_idx':t['idx'],'before':old,'after':t['text']})
            for f in call['facts']:f['source']['span']=None;f['source']['quote']=None
        write(ROOT/f'data/stress96_{level}_input.jsonl',inp(changed));write(ROOT/f'data/stress96_{level}_gold.jsonl',changed)
    write(ROOT/'reports/noise_edits.jsonl',edits)
    manifest={'selection':'24 and 96 evenly spaced midpoint ranks by transcript character count; no gold-based selection','seed':20261115,'curve24_ids':[c['call_id'] for c in small],'stress96_ids':[c['call_id'] for c in stress],'stress_scope':'phonetic noun spelling + fillers; dates/numbers/negations unchanged, gold spans disabled; not a surrogate for all hidden adversarial cases','data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*jsonl')}}
    (ROOT/'reports/checks_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Prepared dev 385, val 383, curve 24, noise 2 x 96')
if __name__=='__main__':main()
