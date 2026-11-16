import subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
raw=json.loads(subprocess.check_output(['docker','image','inspect','cfe-qwen:cpu']))[0]
report={k:raw.get(k) for k in ('Id','RepoTags','Architecture','Os','Size','Created')}
report['model_weights']='read-only mounted models/merged, not included in image size'
(ROOT/'reports/docker_cpu_image.json').write_text(json.dumps(report,indent=2))
freeze=subprocess.check_output(['docker','exec','cfe-qwen-release-check','python','-m','pip','freeze'],text=True)
(ROOT/'reports/docker_cpu_environment.freeze.txt').write_text(freeze)
print(json.dumps(report,indent=2))
