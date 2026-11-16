"""Inference reads input without facts; evaluation is a separate command."""
import argparse,json,time
from pathlib import Path
from collections import Counter
from data import read,write,windows,parse,dedup
from train import prompt_ids

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--limit-calls',type=int,default=0);args=p.parse_args(argv)
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 assert torch.cuda.is_available()
 manifest=json.loads(Path(args.manifest).read_text());tok=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['revision']);tok.pad_token=tok.eos_token
 base=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda')
 model=PeftModel.from_pretrained(base,args.checkpoint).merge_and_unload().eval();model.config.use_cache=True
 calls=read(args.input);assert all('facts' not in c for c in calls.values())
 stats=Counter();pred=[];raw=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 for cid,call in calls.items():
  facts=[]
  for row in windows(call):
   ids,removed=prompt_ids(tok,row);stats['context_lines_removed']+=removed;stats['windows']+=1
   if len(ids)>2048:stats['oversize_input_windows']+=1
   # Do not silently truncate target turns; retain full window up to model capacity.
   if len(ids)>8192:raise ValueError(f'{cid}: window exceeds 8192 tokens; reduce central window size')
   tokens=torch.tensor([ids],device='cuda');mask=torch.ones_like(tokens)
   with torch.inference_mode():output=model.generate(input_ids=tokens,attention_mask=mask,max_new_tokens=640,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   generated=output[0,len(ids):];text=tok.decode(generated,skip_special_tokens=True)
   stats['output_tokens']+=len(generated);stats['generation_limit_hits']+=len(generated)>=640
   parsed,errors=parse(text,call,row['target']);facts.extend(parsed);stats.update(errors);raw.append({'call_id':cid,'target':row['target'],'text':text})
  pred.append({'call_id':cid,'facts':dedup(facts)});stats['calls']+=1
  if stats['calls']%10==0:print(dict(stats),flush=True)
  if args.limit_calls and len(pred)>=args.limit_calls:break
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 write(args.output,pred);write(str(args.output)+'.raw.jsonl',raw)
 stats.update({'wall_seconds':seconds,'seconds_per_call':seconds/len(pred),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'parameters':sum(p.numel() for p in model.parameters())})
 Path(str(args.output)+'.resources.json').write_text(json.dumps(stats,indent=2));print(dict(stats))
if __name__=='__main__':main()
