"""Build measured curve/report only from complete benchmark artifacts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.score import score
from tools.subset_outputs import subset
from tools.compare_outputs import compare

def main(run_dir=None):
 run=Path(run_dir or ROOT/'benchmarks');reports=run/'reports';reports.mkdir(parents=True,exist_ok=True)
 jobs=[('dev_gpu_full','dev'),('val_gpu_full','val'),('curve24_gpu_target','curve24'),('stress96_mild_gpu','stress96_mild'),('stress96_strong_gpu','stress96_strong'),('curve24_cpu','curve24')]
 if (run/'dev_gpu_full.jsonl').exists():
  for name in ('curve24','stress96'):
   subset(run/'dev_gpu_full.jsonl',ROOT/f'data/{name}_input.jsonl',run/f'{name}_gpu_full.jsonl')
   jobs.append((f'{name}_gpu_full',name))
 results={}
 for filename,data in jobs:
  p=run/f'{filename}.jsonl'
  if not p.exists():continue
  try:results[filename]=score(ROOT/f'data/{data}_gold.jsonl',p,reports/f'{filename}.json')
  except (ValueError,AssertionError) as exc:print('INCOMPLETE:',filename,str(exc))
 if 'val_gpu_full' in results:compare(ROOT/'data/val_reference.jsonl',run/'val_gpu_full.jsonl',reports/'val_regression.json')
 if 'dev_gpu_full' in results:compare(ROOT/'reports/dev_checkpoint-2142.jsonl',run/'dev_gpu_full.jsonl',reports/'dev_regression.json')
 lines=['# Измеренные результаты релизной проверки','', '| Режим | Звонки | Temporal | State | Precision | Recall | Grounding turn | Среднее, с | p95, с |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in results.items():
  t=r.get('timing',{});lines.append(f"| {name} | {r['n_calls']} | {100*(r['temporal_recall'] or 0):.2f}% ({r['temporal_correct']}/{r['n_temporal']}) | {100*(r['state_accuracy'] or 0):.2f}% ({r['state_correct']}/{r['n_state']}) | {100*r['fact_precision']:.2f}% | {100*r['fact_recall']:.2f}% | {100*(r['grounding_turn'] or 0):.2f}% | {t.get('mean_s',float('nan')):.3f} | {t.get('p95_s',float('nan')):.3f} |")
 lines+=['','Grounding_span=N/A: спаны не предсказываются. Режимы сравнивать на одинаковом наборе. Память и оборудование — в *.segments.jsonl; для подвыборок GPU память относится к полному исходному прогону. GPU wall-time/1000 — экстраполяция, не время CUDA-ядер.','']
 curve=[(n,results[n]) for n in ('curve24_cpu','curve24_gpu_full','curve24_gpu_target') if n in results and 'timing' in results[n]]
 if len(curve)==3:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(11,4))
  for ax,metric in zip(axes,['temporal_recall','state_accuracy']):
   for name,r in curve:
    ax.scatter(r['timing']['mean_s'],100*r[metric]);ax.annotate(name,(r['timing']['mean_s'],100*r[metric]),fontsize=7,xytext=(3,5),textcoords='offset points')
   ax.set_xscale('log');ax.set_xlabel('Seconds per call (log scale)');ax.set_ylabel(metric+' (%)');ax.grid(alpha=.3)
  fig.suptitle('Same 24 dev calls; CPU full / T4 full / T4 target-only');fig.tight_layout();fig.savefig(reports/'resource_quality.png',dpi=180);fig.savefig(reports/'resource_quality.svg');plt.close(fig)
  lines.append('![Resource-quality](resource_quality.png)')
 else:lines.append(f'Кривая ещё не завершена: доступны {len(curve)} из 3 точек. Недостающие результаты не выдуманы.')
 (reports/'RESULTS.md').write_text('\n'.join(lines));(reports/'all_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else None)
