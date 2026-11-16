"""Incremental per-call benchmark. Resume retains explicit segment metadata."""
import argparse,json,time,platform,threading,sys
from pathlib import Path
from .extractor import Extractor,validate_call

class MemorySampler:
    def __enter__(self):
        import psutil
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.02): self.peak=max(self.peak,self.process.memory_info().rss)
        self.worker=threading.Thread(target=sample,daemon=True);self.worker.start();return self
    def __exit__(self,*args):
        self.stop.set();self.worker.join();self.peak=max(self.peak,self.process.memory_info().rss)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--model-dir');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--dtype',choices=['float16','float32']);p.add_argument('--threads',type=int,default=4);p.add_argument('--context',choices=['full','target_only'],default='full');p.add_argument('--limit',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--warmup',action='store_true');args=p.parse_args(argv)
    calls=[json.loads(s) for s in Path(args.input).read_text().splitlines() if s.strip()]
    if args.limit:calls=calls[:args.limit]
    if not calls:raise ValueError('empty benchmark input')
    if len({c['call_id'] for c in calls})!=len(calls):raise ValueError('duplicate call_id')
    for c in calls:validate_call(c)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not args.resume:raise FileExistsError(f'{out} exists; choose another output or --resume')
    done={json.loads(s)['call_id'] for s in out.read_text().splitlines()} if out.exists() else set()
    if not done.issubset({c['call_id'] for c in calls}):raise ValueError('resume output has unexpected call ids')
    extractor=Extractor(args.model_dir,args.device,args.dtype,args.threads,args.context)
    torch=extractor.torch
    import transformers,peft,hashlib
    model_meta=Path(args.model_dir or Path(__file__).resolve().parents[1]/'models/merged')/'release_manifest.json'
    config_digest=hashlib.sha256(json.dumps({'inference':extractor.config,'device':args.device,'dtype':extractor.dtype,'context':args.context,'threads':args.threads,'input_sha':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'model_manifest':model_meta.read_text() if model_meta.exists() else None,'torch':torch.__version__,'transformers':transformers.__version__,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},sort_keys=True).encode()).hexdigest()
    metadata=Path(str(out)+'.segments.jsonl')
    run_config=Path(str(out)+'.run_config.json')
    if run_config.exists() and json.loads(run_config.read_text())['config_digest']!=config_digest:raise ValueError('resume run configuration mismatch')
    run_config.write_text(json.dumps({'config_digest':config_digest},indent=2))
    if done:
        timing_path=Path(str(out)+'.timings.jsonl')
        timing_ids=[json.loads(s)['call_id'] for s in timing_path.read_text().splitlines()]
        if len(timing_ids)!=len(done) or set(timing_ids)!=done:raise ValueError('Interrupted write detected: use a fresh output path; incomplete output must not be silently resumed')
    if args.resume and metadata.exists():
        previous=[json.loads(s) for s in metadata.read_text().splitlines()]
        if any(r['config_digest']!=config_digest for r in previous):raise ValueError('resume configuration mismatch')
    if len(done)==len(calls):print('All calls already completed, configuration verified',flush=True);return
    warmup_seconds=None
    if args.warmup:
        print('Warmup: first call (excluded from benchmark)',flush=True);_,stat,_,_=extractor.extract(calls[0]);warmup_seconds=stat['seconds']
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    total=0;started=time.perf_counter()
    with out.open('a') as pred,Path(str(out)+'.timings.jsonl').open('a') as timings,Path(str(out)+'.raw.jsonl').open('a') as rawfile,MemorySampler() as memory:
        for i,call in enumerate(calls):
            if call['call_id'] in done:continue
            result,stats,raw,partial=extractor.extract(call)
            stat={'call_id':call['call_id'],**stats,'partial':partial,'turns':len(call['turns']),'facts':len(result['facts']),'cold_first_call':total==0 and not args.warmup}
            pred.write(json.dumps(result,ensure_ascii=False)+'\n');pred.flush()
            timings.write(json.dumps(stat)+'\n');timings.flush()
            for r in raw:rawfile.write(json.dumps(r,ensure_ascii=False)+'\n')
            rawfile.flush();total+=1
            print(f"[{i+1}/{len(calls)}] {call['call_id']} {stats['seconds']:.3f}s facts={len(result['facts'])} partial={partial}",flush=True)
    segment={'config_digest':config_digest,'calls_this_segment':total,'loop_wall_seconds':time.perf_counter()-started,'load_seconds':extractor.load_seconds,'warmup_seconds':warmup_seconds,'peak_rss_bytes':memory.peak,'rss_sampling_interval_ms':20,'parameters':extractor.parameters,'device':args.device,'dtype':extractor.dtype,'context':args.context,'threads':args.threads,'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if args.device=='cuda' else None}
    with metadata.open('a') as stream:stream.write(json.dumps(segment)+'\n')
    print(json.dumps(segment,indent=2),flush=True)
if __name__=='__main__':main()
