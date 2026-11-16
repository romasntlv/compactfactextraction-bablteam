"""Synthetic TRAIN-only contrastive curriculum with facts fixed before rendering."""
import json,random,copy,hashlib
from pathlib import Path
from datetime import date,timedelta
OUT=Path(__file__).resolve().parent

def generate():
 rows=[];rng=random.Random(20261115)
 for group in range(100):
  anchor=date(2026,1,1)+timedelta(days=rng.randrange(350));ticket=str(10000+group);other=str(20000+group)
  cases=[
   ('offer_accepted',[('operator','Специалист может приехать завтра. Вам подходит?'),('client','Да, подходит, договорились.')],[(1,'agreement','выезд специалиста','клиент','client',1,1,None)]),
   ('offer_rejected',[('operator','Специалист может приехать завтра. Вам подходит?'),('client','Нет, не подходит. Пока ничего не назначайте.')],[]),
   ('guess_corrected',[('operator','У вас сейчас тариф «Базовый»?'),('client','Да нет, сейчас у меня «Плюс». Когда он появился, не помню.')],[(1,'tariff','Плюс','клиент','client',1,0,None)]),
   ('guess_confirmed',[('operator','У вас сейчас тариф «Базовый»?'),('client','Да, всё верно, сейчас «Базовый». Дату подключения не помню.')],[(1,'tariff','Базовый','клиент','client',1,0,None)]),
   ('hypothesis_not_transition',[('operator',f'Заявка {ticket} принята позавчера. Если завтра её выполнят, статус будет «выполнена». Пока она только принята.')],[(0,'ticket_status','принята','заявка '+ticket,'ticket',1,-2,None)]),
   ('actual_transition',[('operator',f'Заявка {ticket} принята позавчера. Вчера она перешла в статус «в работе».')],[(0,'ticket_status','принята','заявка '+ticket,'ticket',1,-2,-1),(0,'ticket_status','в работе','заявка '+ticket,'ticket',1,-1,None)]),
   ('visit_is_not_status_date',[('operator',f'Заявка {ticket} в работе со вчерашнего дня. Предлагаю выезд специалиста завтра. Подходит?'),('client','Да, завтра подходит.')],[(0,'ticket_status','в работе','заявка '+ticket,'ticket',1,-1,None),(1,'agreement','выезд специалиста','клиент','client',1,1,None)]),
   ('different_tickets_and_third_party',[('client',f'Сосед говорит, что его заявка {other} выполнена. Что с моей заявкой {ticket}?'),('operator',f'Ваша заявка {ticket} принята сегодня. Статус заявки соседа я не подтверждаю.')],[(1,'ticket_status','принята','заявка '+ticket,'ticket',1,0,None)])]
  for kind,turns,plan in cases:
   for noisy in (False,True):
    cid=f'synth_schema_{group:04d}_{kind}_{int(noisy)}';rendered=[]
    for i,(speaker,text) in enumerate(turns):
     if noisy:
      text='Эээ, '+text.replace('Специалист','Спецыалист').replace('специалиста','спецыалиста').replace('Заявка','Заяфка').replace('заявка','заяфка').replace('подключения','подклющения')
     rendered.append({'idx':i*3,'speaker':speaker,'text':text})
    facts=[]
    for pos,attr,value,name,typ,pol,lo,hi in plan:
     t=rendered[pos];facts.append({'object':{'id':cid+':'+name,'name':name,'type':typ},'attribute':attr,'value':value,'polarity':pol,'speaker':t['speaker'],'valid_from':(anchor+timedelta(days=lo)).isoformat()+'T00:00:00','valid_to':(anchor+timedelta(days=hi)).isoformat()+'T00:00:00' if hi is not None else None,'source':{'turn_idx':t['idx'],'span':None,'quote':t['text']}})
    rows.append({'call_id':cid,'started_at':anchor.isoformat()+'T12:00:00','series':{'account_id':f'synth_schema_{group:04d}','call_index':0,'series_len':1},'turns':rendered,'facts':facts,'synthetic_metadata':{'family':kind,'contrast_group':group,'noisy':noisy,'train_only':True,'generator_seed':20261115}})
 return rows

def validate(rows):
 assert len(rows)==1600 and len({r['call_id'] for r in rows})==len(rows)
 by_id={r['call_id']:r for r in rows}
 for row in rows:
  turns={t['idx']:t for t in row['turns']}
  for f in row['facts']:
   assert f['source']['quote']==turns[f['source']['turn_idx']]['text']
   assert not f['valid_to'] or f['valid_from']<=f['valid_to']
  kind=row['synthetic_metadata']['family']
  if kind=='offer_rejected':assert row['facts']==[]
  if kind=='hypothesis_not_transition':assert len(row['facts'])==1 and row['facts'][0]['valid_to'] is None
  if kind=='actual_transition':assert row['facts'][0]['valid_to']==row['facts'][1]['valid_from']
  if kind=='visit_is_not_status_date':assert row['facts'][0]['valid_from']<row['started_at']<row['facts'][1]['valid_from']
  if kind=='different_tickets_and_third_party':assert len(row['facts'])==1 and row['facts'][0]['object']['name'].endswith(str(10000+row['synthetic_metadata']['contrast_group']))
  if kind=='guess_corrected':assert row['facts'][0]['value']=='Плюс' and row['facts'][0]['polarity']==1
  if row['synthetic_metadata']['noisy']:
   clean=by_id[row['call_id'][:-1]+'0'];assert [(f['attribute'],f['value'],f['valid_from'],f['valid_to']) for f in clean['facts']]==[(f['attribute'],f['value'],f['valid_from'],f['valid_to']) for f in row['facts']]

if __name__=='__main__':
 rows=generate();validate(rows);p=OUT/'synthetic_train.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (OUT/'synthetic_manifest.json').write_text(json.dumps({'calls':len(rows),'families':8,'contrast_groups':100,'empty_calls':sum(not r['facts'] for r in rows),'seed':20261115,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'train-only pilot; structural and contrast tests passed; training efficacy unmeasured'},indent=2));print('1600 synthetic train calls; contrast/source/date invariants PASS')
