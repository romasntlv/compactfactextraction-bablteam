import os,logging
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Response
from .extractor import Extractor,validate_call,GenerationError

def create_app(factory=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.extractor=(factory or (lambda:Extractor(os.environ.get('MODEL_DIR'),os.environ.get('DEVICE','cpu'),os.environ.get('DTYPE'),int(os.environ.get('CPU_THREADS','4')))))()
        yield
    app=FastAPI(title='CompactFactExtraction Team 3',version='1.0.0',lifespan=lifespan)
    @app.get('/health')
    def health():return {'status':'ready','model_loaded':hasattr(app.state,'extractor')}
    @app.post('/extract')
    def extract(call:dict,response:Response):
        try:validate_call(call)
        except ValueError as exc:raise HTTPException(422,str(exc))
        try:result,stats,raw,partial=app.state.extractor.extract(call)
        except GenerationError as exc:raise HTTPException(502,str(exc))
        except Exception:
            logging.exception('Inference failure');raise HTTPException(500,'Inference failed')
        response.headers['X-CFE-Partial']='true' if partial else 'false'
        response.headers['X-CFE-Model-Seconds']=str(stats['seconds'])
        response.headers['X-CFE-Invalid-Rows']=str(stats.get('invalid_rows',0))
        response.headers['X-CFE-Invalid-JSON']=str(stats.get('invalid_json',0))
        response.headers['X-CFE-Limit-Hits']=str(stats.get('generation_limit_hits',0))
        return result
    return app
app=create_app()
