"""Clone F03's exact frozen code and initial all-source pose; default no physics."""
from pathlib import Path
import argparse,shutil,subprocess,sys
from datetime import datetime,timedelta,timezone

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/unified_curve_transfer_20261007/FULL_SOURCE_INNER_20261009'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--case',default='F05',choices=['F01','F02','F03','F04','F05']);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    source=BASE/args.case;out=Path(args.out).resolve()
    if out.exists():raise FileExistsError(out)
    shutil.copytree(source/'sources',out)
    shutil.copy2(source/'initial_pose.npz',out/'initial_pose.npz')
    deadline=(datetime.now(timezone.utc)+timedelta(seconds=1100)).isoformat()
    command=[sys.executable,str(out/'validation/full_source_inner_experiment.py'),'--name','REPRODUCE_'+args.case,'--initial',str(out/'initial_pose.npz'),'--max-wall','800','--deadline',deadline]
    if args.case in ['F02','F03']:command.append('--anchored-probe')
    if args.case in ['F04','F05']:command.append('--braced')
    if args.case=='F05':command.append('--polar-unfold')
    if not args.run:command.append('--prepare-only')
    print('Default reproduction prepares geometry with zero physical steps; --run replays one new bounded actual trial.',flush=True)
    subprocess.run(command,cwd=out,check=True)


if __name__=='__main__':main()
