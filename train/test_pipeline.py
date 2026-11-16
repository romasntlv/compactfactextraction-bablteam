import json,tempfile
from pathlib import Path
from data import windows,label,parse,dedup,ROOT,read

def unit():
 call={'call_id':'test:0','started_at':'2026-05-20T12:00:00','turns':[{'idx':i*3,'speaker':'operator','text':f'Заявка {1234+i} принята вчера.'} for i in range(15)]}
 facts=[{'object':{'name':'заявка 1234','type':'ticket'},'attribute':'ticket_status','value':'принята','polarity':1,'valid_from':'2026-05-19','valid_to':None,'source':{'turn_idx':0}}]
 ws=list(windows(call));assert sorted(i for w in ws for i in w['target'])==[t['idx'] for t in call['turns']]
 answer=label({**call,'facts':facts},ws[0]);parsed,errors=parse(answer,call,ws[0]['target']);assert len(parsed)==1 and parsed[0]['valid_from']=='2026-05-19';assert parsed[0]['source']['turn_idx']==0
 assert parse('not json',call,[0])[0]==[]
 assert parse('[[999,"K","заявка 1","ticket","принята",1,0,null]]',call,[0])[0]==[]
 assert parse('[[0,"K","заявка 1","ticket","принята",1,0,-1]]',call,[0])[0]==[]
 assert len(dedup(parsed+parsed))==1
 assert parse('[]',call,[0])[0]==[]
 print('codec/window/source/interval/empty tests: PASS')

def tiny_backward():
 import torch
 from transformers import Qwen2Config,Qwen2ForCausalLM
 from train import collator
 class Tok:pad_token_id=0
 model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2))
 batch=collator(Tok())([{'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]},{'input_ids':[1,2,5],'labels':[-100,-100,5]}])
 assert batch['labels'][1,-1].item()==-100
 result=model(**batch);assert torch.isfinite(result.loss);result.loss.backward();assert any(p.grad is not None for p in model.parameters());print('tiny random Qwen CPU forward/backward: PASS')
if __name__=='__main__':
 unit()
 import sys
 if '--backward' in sys.argv:tiny_backward()
