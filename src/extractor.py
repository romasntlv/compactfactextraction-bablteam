"""One offline model, shared by CLI and HTTP. No date_extractor or model cascade."""
import json, time, threading
from pathlib import Path
from collections import Counter
from datetime import datetime
from .codec import windows, parse, dedup
ROOT=Path(__file__).resolve().parents[1]

class GenerationError(RuntimeError):
    pass

def validate_call(call):
    if not isinstance(call,dict): raise ValueError('call must be an object')
    if 'facts' in call: raise ValueError('Input facts are forbidden; supply transcript and metadata only')
    if not isinstance(call.get('call_id'),str) or not call['call_id']: raise ValueError('call_id must be a nonempty string')
    try: datetime.fromisoformat(call['started_at'])
    except (KeyError,TypeError,ValueError): raise ValueError('started_at must be ISO datetime')
    if not isinstance(call.get('turns'),list): raise ValueError('turns must be a list')
    seen=set()
    for turn in call['turns']:
        if not isinstance(turn,dict) or type(turn.get('idx')) is not int: raise ValueError('turn idx must be integer')
        if turn['idx'] in seen: raise ValueError('duplicate turn idx')
        seen.add(turn['idx'])
        if turn.get('speaker') not in ('client','operator'): raise ValueError('speaker must be client or operator')
        if not isinstance(turn.get('text'),str): raise ValueError('turn text must be string')

def validate_facts(call,facts):
    turns={t['idx']:t for t in call['turns']}
    for f in facts:
        if f['object']['type'] not in ('service','device','ticket','client'): raise ValueError('invalid object type')
        if f['attribute'] not in ('service_state','tariff','ticket_status','action','agreement'): raise ValueError('invalid attribute')
        if not isinstance(f['object']['name'],str) or not isinstance(f['value'],str): raise ValueError('invalid strings')
        if type(f['polarity']) is not int or f['polarity'] not in (-1,1): raise ValueError('invalid polarity')
        start=datetime.fromisoformat(f['valid_from'])
        if f['valid_to'] is not None and datetime.fromisoformat(f['valid_to'])<start: raise ValueError('inverted interval')
        t=turns[f['source']['turn_idx']]
        if f['source']['quote']!=t['text'] or f['speaker']!=t['speaker']: raise ValueError('invalid grounding')
        if f['source']['span'] is not None: raise ValueError('this release emits null spans')

def prompt_ids(tokenizer,row,max_input=1536):
    # Identical to training prompt construction.
    user=row['user'];removed=0
    def encode(): return tokenizer.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':user}],tokenize=True,add_generation_prompt=True)
    tokens=encode()
    while len(tokens)>max_input:
        lines=user.splitlines();context=[i for i,s in enumerate(lines) if s.startswith('[') and int(s[1:].split()[0]) not in row['target']]
        if not context: break
        del lines[context[-1]];user='\n'.join(lines);removed+=1;tokens=encode()
    return tokens,removed

class Extractor:
    def __init__(self,model_dir=None,device='cpu',dtype=None,threads=4,context='full'):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.device=device;self.context=context;self.lock=threading.Lock()
        if context not in ('full','target_only'): raise ValueError('invalid context mode')
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(threads);torch.manual_seed(42)
        self.dtype=dtype or ('float16' if device=='cuda' else 'float32')
        self.config=json.loads((ROOT/'configs/inference.json').read_text())
        start=time.perf_counter();path=Path(model_dir or ROOT/'models/merged')
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=getattr(torch,self.dtype),attn_implementation='sdpa').to(device).eval()
        self.model.config.use_cache=True
        self.sync();self.load_seconds=time.perf_counter()-start
        self.parameters=sum(p.numel() for p in self.model.parameters())
    def sync(self):
        if self.device=='cuda': self.torch.cuda.synchronize()
    def extract(self,call):
        validate_call(call)
        with self.lock: return self._extract(call)
    def _extract(self,call):
        torch=self.torch;stats=Counter();facts=[];raw=[]
        self.sync();start=time.perf_counter()
        for row in windows(call,central=self.config['central'],radius=self.config['radius']):
            if self.context=='target_only':
                lines=row['user'].splitlines()
                row={**row,'user':'\n'.join(s for s in lines if not s.startswith('[') or int(s[1:].split()[0]) in row['target'])}
            ids,removed=prompt_ids(self.tokenizer,row,self.config['max_input'])
            stats['context_lines_removed']+=removed;stats['windows']+=1;stats['input_tokens']+=len(ids)
            if len(ids)>self.config['max_window_tokens']: raise GenerationError('input window exceeds configured token limit')
            tokens=torch.tensor([ids],device=self.device)
            with torch.inference_mode():
                output=self.model.generate(input_ids=tokens,attention_mask=torch.ones_like(tokens),max_new_tokens=self.config['max_new_tokens'],do_sample=False,pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id)
            generated=output[0,len(ids):];text=self.tokenizer.decode(generated,skip_special_tokens=True)
            stats['output_tokens']+=len(generated);stats['generation_limit_hits']+=len(generated)>=self.config['max_new_tokens']
            parsed,errors=parse(text,call,row['target']);stats.update(errors)
            # Legacy codec allowed 'other'; public schema does not. Reject explicitly and count.
            for fact in parsed:
                if fact['object']['type']=='other': stats['invalid_rows']+=1
                else: facts.append(fact)
            raw.append({'call_id':call['call_id'],'target':row['target'],'text':text})
        facts=dedup(facts);validate_facts(call,facts);self.sync()
        stats['seconds']=time.perf_counter()-start
        partial=any(stats[k] for k in ('invalid_json','invalid_rows','generation_limit_hits'))
        # A wholly unparseable response is an error, never a successful empty extraction.
        if stats['windows'] and stats['invalid_json']==stats['windows']: raise GenerationError('all model windows returned malformed JSON')
        return {'call_id':call['call_id'],'facts':facts},dict(stats),raw,partial
