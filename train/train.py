"""LoRA SFT, assistant-only loss, FP16/T4, immutable train/dev account split."""
import os,json,argparse,time
from pathlib import Path
from data import ROOT
MODEL='Qwen/Qwen2.5-0.5B-Instruct'

def prompt_ids(tokenizer,row,max_input=1536):
 user=row['user'];removed=0
 def encode():return tokenizer.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':user}],tokenize=True,add_generation_prompt=True)
 tokens=encode()
 # Remove only context, never central target turns. Same function at inference.
 while len(tokens)>max_input:
  lines=user.splitlines();context=[i for i,s in enumerate(lines) if s.startswith('[') and int(s[1:].split()[0]) not in row['target']]
  if not context:break
  del lines[context[-1]];user='\n'.join(lines);removed+=1;tokens=encode()
 return tokens,removed

class Dataset:
 def __init__(self,path,tokenizer,max_length=2048,limit=0):
  self.rows=[];self.stats={'source':0,'skipped_oversize':0,'context_lines_removed':0}
  from tqdm.auto import tqdm
  for line in tqdm(Path(path).read_text().splitlines(),desc='Токенизация '+Path(path).name):
   row=json.loads(line);self.stats['source']+=1;prompt,removed=prompt_ids(tokenizer,row)
   answer=tokenizer.encode(row['answer']+tokenizer.eos_token,add_special_tokens=False)
   self.stats['context_lines_removed']+=removed
   if len(prompt)+len(answer)>max_length:self.stats['skipped_oversize']+=1;continue
   self.rows.append({'input_ids':prompt+answer,'labels':[-100]*len(prompt)+answer})
   if limit and len(self.rows)>=limit:break
  assert self.rows,'No training samples fit max_length'
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.rows[i]

def collator(tokenizer):
 import torch
 def collate(rows):
  n=((max(len(x['input_ids']) for x in rows)+7)//8)*8
  return {'input_ids':torch.tensor([r['input_ids']+[tokenizer.pad_token_id]*(n-len(r['input_ids'])) for r in rows]),'attention_mask':torch.tensor([[1]*len(r['input_ids'])+[0]*(n-len(r['input_ids'])) for r in rows]),'labels':torch.tensor([r['labels']+[-100]*(n-len(r['labels'])) for r in rows])}
 return collate

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'run'));parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--max-steps',type=int,default=-1);parser.add_argument('--resume',default=None);parser.add_argument('--train-data',type=Path,default=ROOT/'prepared/train.jsonl');args=parser.parse_args(argv)
 import torch,transformers,peft
 from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,Trainer,TrainerCallback,set_seed
 from peft import LoraConfig,get_peft_model
 from huggingface_hub import HfApi
 assert torch.cuda.is_available(),'Select a GPU runtime (T4) in Colab'
 set_seed(42);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 manifest_path=out/'model_manifest.json'
 revision=json.loads(manifest_path.read_text())['revision'] if manifest_path.exists() else '7ae557604adf67be50417f59c2c2f167def9a775'
 tok=AutoTokenizer.from_pretrained(MODEL,revision=revision);tok.pad_token=tok.eos_token;tok.padding_side='right'
 model=AutoModelForCausalLM.from_pretrained(MODEL,revision=revision,torch_dtype=torch.float16,attn_implementation='sdpa')
 n_base=sum(p.numel() for p in model.parameters());assert n_base<1_500_000_000
 model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
 model.enable_input_require_grads();model.config.use_cache=False
 training=Dataset(args.train_data,tok,limit=args.limit)
 # Loss monitor only; checkpoint selection is by decoded full-dev official metrics.
 dev=Dataset(ROOT/'prepared/dev.jsonl',tok,limit=64 if args.limit else 0)
 manifest={'model':MODEL,'revision':revision,'base_parameters':n_base,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name(),'train':training.stats,'dev':dev.stats,'seed':42,'train_data':str(args.train_data),'train_sha256':__import__('hashlib').sha256(args.train_data.read_bytes()).hexdigest()}
 manifest_path.write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)
 class LiveStatus(TrainerCallback):
  def on_epoch_begin(self,args,state,control,**kwargs):
   print(f'\n▶ Эпоха {int(state.epoch or 0)+1}/{args.num_train_epochs:g}; шаг {state.global_step}/{state.max_steps}',flush=True)
  def on_log(self,args,state,control,logs=None,**kwargs):
   record={'epoch':state.epoch,'step':state.global_step,**(logs or {})}
   print(json.dumps(record,ensure_ascii=False),flush=True)
   with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
  def on_save(self,args,state,control,**kwargs):
   print(f'💾 Сохранён checkpoint-{state.global_step}',flush=True)
  def on_train_end(self,args,state,control,**kwargs):
   print('✅ Обучение завершено',flush=True)
 config=TrainingArguments(output_dir=str(out),num_train_epochs=args.epochs,max_steps=args.max_steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=16,learning_rate=2e-4,lr_scheduler_type='cosine',warmup_ratio=.05,fp16=True,bf16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='epoch',save_strategy='epoch',save_total_limit=3,logging_steps=10,logging_first_step=True,disable_tqdm=False,report_to='none',optim='adamw_torch',seed=42,data_seed=42,dataloader_num_workers=0,remove_unused_columns=False)
 trainer=Trainer(model=model,args=config,train_dataset=training,eval_dataset=dev,data_collator=collator(tok),processing_class=tok,callbacks=[LiveStatus()])
 start=time.perf_counter();trainer.train(resume_from_checkpoint=args.resume);elapsed=time.perf_counter()-start
 trainer.save_model(str(out/'final'));tok.save_pretrained(out/'final')
 (out/'training_resources.json').write_text(json.dumps({'train_seconds':elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2))
if __name__=='__main__':main()
