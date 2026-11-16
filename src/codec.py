"""Gold-free window builder and strict compact JSON codec."""
import json,re,sys
from pathlib import Path
from datetime import date,timedelta
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'evaluation'))
import case_metrics as metrics
CODES={'service_state':'S','tariff':'T','ticket_status':'K','action':'A','agreement':'G'}
ATTRS={v:k for k,v in CODES.items()}
SYSTEM='''Извлеки подтвержденные факты звонка поддержки. Верни только JSON-массив строк: [номер_реплики,атрибут,объект,тип_объекта,значение,полярность,начало,конец]. Атрибуты: S=service_state,T=tariff,K=ticket_status,A=action,G=agreement. Типы: service,device,ticket,client. Даты — целое число дней относительно даты звонка; конец null означает действует дальше. Неизвестное начало=0. Полярность -1 только явное отрицание использования клиентом. Предложения без согласия и отвергнутые догадки не факты. Каждая заявка имеет свою историю. Дата визита не является датой статуса. Выводи факты только из целевых реплик; соседние реплики даны для контекста. Нет фактов: [].'''

def read(path):
 rows={}
 for line in Path(path).read_text().splitlines():
  r=json.loads(line);assert r['call_id'] not in rows;rows[r['call_id']]=r
 return rows

def write(path,rows):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def recover(text,anchor):
 # Official 2.1 contains out.update(date,date,...) in three rare branches.
 # Do not modify evaluation; protect only optional prompt annotations.
 try:return metrics.recover_dates(text,anchor)
 except TypeError:return set()

def windows(call,central=6,radius=3):
 assert 'facts' not in call,'Inference window builder must not receive gold'
 turns=call['turns'];anchor=date.fromisoformat(call['started_at'][:10])
 for start in range(0,len(turns),central):
  stop=min(len(turns),start+central);target=[t['idx'] for t in turns[start:stop]]
  surrounding=list(range(max(0,start-radius),min(len(turns),stop+radius)))
  # Distant explicit ticket/date evidence, text-only and bounded. Never target it twice.
  numbers=set(re.findall(r'\b\d{3,6}\b',' '.join(t['text'] for t in turns[start:stop])))
  distant=[i for i,t in enumerate(turns) if i not in surrounding and numbers.intersection(re.findall(r'\b\d{3,6}\b',t['text']))]
  distant=sorted(distant,key=lambda i:abs(i-start))[:2]
  header=f"Дата звонка: {anchor.isoformat()}, день недели {anchor.weekday()} (0=понедельник). Целевые реплики: {target}.\n"
  lines=[]
  for i in sorted(surrounding+distant):
   t=turns[i];ds=sorted((d-anchor).days for d in recover(t['text'],anchor))
   lines.append(f"[{t['idx']} {t['speaker']} даты={ds}] {t['text']}")
  yield {'call_id':call['call_id'],'target':target,'system':SYSTEM,'user':header+'\n'.join(lines)}

def label(call,window):
 anchor=date.fromisoformat(call['started_at'][:10]);rows=[]
 for f in call['facts']:
  if f['source']['turn_idx'] not in window['target']:continue
  def offset(v):return (date.fromisoformat(v[:10])-anchor).days if v else None
  rows.append([f['source']['turn_idx'],CODES[f['attribute']],f['object']['name'],f['object']['type'],f['value'],f['polarity'],offset(f['valid_from']),offset(f['valid_to'])])
 return json.dumps(rows,ensure_ascii=False,separators=(',',':'))

def parse(text,call,target):
 text=text.strip()
 if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
 try:rows=json.loads(text)
 except (ValueError,TypeError):return [],{'invalid_json':1}
 if not isinstance(rows,list):return [],{'invalid_json':1}
 turns={t['idx']:t for t in call['turns']};anchor=date.fromisoformat(call['started_at'][:10]);facts=[];rejected=0
 for r in rows:
  if not isinstance(r,list) or len(r)!=8:rejected+=1;continue
  idx,attr,name,kind,value,polarity,start,end=r
  if type(idx)!=int or idx not in target or idx not in turns or not isinstance(attr,str) or attr not in ATTRS or not isinstance(name,str) or not isinstance(value,str) or not isinstance(kind,str) or kind not in ('service','device','ticket','client','other') or type(polarity)!=int or polarity not in (-1,1) or type(start)!=int or not -3660<=start<=3660 or (end is not None and (type(end)!=int or not -3660<=end<=3660 or end<start)):
   rejected+=1;continue
  t=turns[idx];facts.append({'object':{'name':name,'type':kind},'attribute':ATTRS[attr],'value':value,'polarity':polarity,'speaker':t['speaker'],'valid_from':(anchor+timedelta(days=start)).isoformat(),'valid_to':(anchor+timedelta(days=end)).isoformat() if end is not None else None,'source':{'turn_idx':idx,'span':None,'quote':t['text']}})
 return facts,{'invalid_rows':rejected}

def dedup(facts):
 seen=set();out=[]
 for f in facts:
  key=(f['object']['name'],f['attribute'],f['value'],f['polarity'],f['valid_from'],f['valid_to'])
  if key not in seen:seen.add(key);out.append(f)
 return out
