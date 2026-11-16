"""Reproduce original FP16 merge on CPU; save once for offline CPU/GPU serving."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 target=ROOT/'models/merged'
 if (target/'release_manifest.json').exists():
  manifest=json.loads((target/'release_manifest.json').read_text())
  for name,digest in manifest['sha256'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
  print('Merged model already present; checksums verified');return
 torch.set_num_threads(4)
 base=AutoModelForCausalLM.from_pretrained(ROOT/'models/base',local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa')
 model=PeftModel.from_pretrained(base,ROOT/'models/adapter').merge_and_unload().eval()
 model.save_pretrained(target,safe_serialization=True)
 tok=AutoTokenizer.from_pretrained(ROOT/'models/base',local_files_only=True);tok.pad_token=tok.eos_token;tok.save_pretrained(target)
 manifest={'base_model':'Qwen/Qwen2.5-0.5B-Instruct','revision':'7ae557604adf67be50417f59c2c2f167def9a775','checkpoint':'checkpoint-2142','merge_dtype':'float16','merge_device':'cpu','parameters':sum(p.numel() for p in model.parameters()),'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()}}
 (target/'release_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
