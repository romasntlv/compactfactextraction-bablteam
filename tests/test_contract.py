import unittest,copy,json
from pathlib import Path
from src.codec import windows,parse
from src.extractor import validate_call,validate_facts,prompt_ids,GenerationError
from src.api import create_app
from fastapi.testclient import TestClient
CALL={'call_id':'test','started_at':'2026-05-20T12:00:00','turns':[{'idx':3,'speaker':'operator','text':'Заявка 1234 принята позавчера, вчера перешла в работу.'},{'idx':8,'speaker':'client','text':'Договорились, специалист приедет завтра.'}]}
class Fake:
 def extract(self,c):return {'call_id':c['call_id'],'facts':[]},{'seconds':.01},[],False
class Broken:
 def extract(self,c):raise GenerationError('malformed JSON')
class Contract(unittest.TestCase):
 def test_idx_and_dates(self):
  rows=[[3,'K','заявка 1234','ticket','принята',1,-2,-1],[3,'K','заявка 1234','ticket','в работе',1,-1,None],[8,'G','клиент','client','выезд специалиста',1,1,None]]
  facts,err=parse(json.dumps(rows),CALL,[3,8]);validate_facts(CALL,facts)
  self.assertEqual(facts[0]['valid_to'],facts[1]['valid_from']);self.assertEqual(facts[2]['valid_from'],'2026-05-21');self.assertEqual(facts[2]['speaker'],'client')
 def test_independent_tickets_and_same_day(self):
  rows=[[3,'K','заявка 1','ticket','принята',1,0,0],[3,'K','заявка 2','ticket','в работе',1,-1,None]]
  facts,err=parse(json.dumps(rows),CALL,[3]);validate_facts(CALL,facts);self.assertEqual(len(facts),2)
 def test_bad_intervals_and_indices(self):
  for row in [[9,'K','x','ticket','принята',1,0,None],[3,'K','x','ticket','принята',1,0,-1]]:self.assertEqual(parse(json.dumps([row]),CALL,[3])[1]['invalid_rows'],1)
 def test_input_invalid(self):
  for c in [{**CALL,'facts':[]},{**CALL,'started_at':'bad'},{**CALL,'turns':CALL['turns']*2}]:
   with self.assertRaises(ValueError):validate_call(c)
 def test_window_coverage(self):
  c={**CALL,'turns':[{'idx':i*3,'speaker':'client','text':'да'} for i in range(20)]}
  self.assertEqual([i for w in windows(c) for i in w['target']],[t['idx'] for t in c['turns']])
 def test_api(self):
  with TestClient(create_app(Fake)) as client:
   self.assertEqual(client.get('/health').status_code,200)
   r=client.post('/extract',json=CALL);self.assertEqual(r.json(),{'call_id':'test','facts':[]});self.assertEqual(r.headers['X-CFE-Partial'],'false')
   self.assertEqual(client.post('/extract',json={**CALL,'facts':[]}).status_code,422)
 def test_api_error_not_empty_success(self):
  with TestClient(create_app(Broken)) as client:self.assertEqual(client.post('/extract',json=CALL).status_code,502)
 def test_empty_turns_allowed(self):validate_call({**CALL,'turns':[]})
 def test_prompt_matches_training(self):
  import ast
  source=Path(__file__).resolve().parents[1]/'train/train.py'
  node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prompt_ids')
  namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
  class Tok:
   def apply_chat_template(self,messages,**kwargs):return list(json.dumps(messages,ensure_ascii=False).encode())
  row={'system':'test','target':[3],'user':'Дата: 2026-05-20\n[0 operator] контекст\n[3 client] целевая реплика\n[8 operator] контекст'}
  self.assertEqual(prompt_ids(Tok(),row,100),namespace['prompt_ids'](Tok(),row,100))

if __name__=='__main__':unittest.main()
