"""Wait within the active task, then invoke the unchanged exact-version runner once."""
from pathlib import Path
import argparse,json,time,sys,subprocess
ap=argparse.ArgumentParser();ap.add_argument('branch');args=ap.parse_args();P=Path(__file__).resolve().parent;B=(P/args.branch).resolve()
if B.parent!=P:raise ValueError('Expected a direct child controller branch')
if (B/'T_UPDATE').exists():raise RuntimeError('Case already exists; never start a duplicate')
version=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'))['version'];print('Waiting for independent approval of '+version+'; zero integration by this waiter.',flush=True)
while True:
    if (B/'cancel_wait.request').exists():raise SystemExit('Wait cancelled without starting dynamics')
    try:
        r=json.loads((B/'REGISTERED.json').read_text(encoding='utf-8'));a=json.loads((B/'INDEPENDENT_CONTROLLER_REVIEW.json').read_text(encoding='utf-8'))
        ready=r['version']==version and all(r.get(k) is True for k in ['dynamics_authorized','independent_initial_and_chain_review_passed','current_code_independently_reviewed']) and a.get('approved') is True and a.get('version')==version and 'T_UPDATE' in a.get('approved_case_names',[])
    except (FileNotFoundError,json.JSONDecodeError):ready=False
    if ready:break
    time.sleep(45)
if (B/'T_UPDATE').exists():raise RuntimeError('Case appeared while waiting; no duplicate start')
print('Current independent gate open; invoke frozen runner, which independently verifies exact code, initial, registration and physical assets.',flush=True)
raise SystemExit(subprocess.run([sys.executable,'-u',str(B/'direct_runner.py'),'--case','T_UPDATE']).returncode)
