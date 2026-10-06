"""Finalize saved action records only; no physics, hashing or repository audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import json
import csv
import re
import shutil
import numpy as np
import cv2
from target_open_close_geometry import BASE,PARENT,write


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    history=read_rows(BASE/'historical_15_trial_ledger.csv')
    total=read_rows(PARENT/'trial_ledger.csv');local=read_rows(BASE/'trial_ledger.csv')
    if len(total)!=17 or len(local)!=2 or total[:15]!=history or total[15:]!=local:
        raise RuntimeError('History/local/total inconsistency; do not close')
    registry=json.loads((BASE/'revision_registry.json').read_text(encoding='utf8'))
    registry.update(status='STOPPED',new_dynamics=2,total_dynamics=17,further_dynamics_authorized=False,
        actual_open_close='OBSERVED_A2',post_close_grasp='FAIL',original_late_failure_interval='NOT_REACHED',
        remaining_unused_trials=2,reason='Actual opening and closing executed; target outside post-close grasp; contact screen fails')
    write(BASE/'revision_registry.json',registry)
    a=np.load(BASE/'OPEN_CLOSE_A1_ACTION/trace.npz');b=np.load(BASE/'OPEN_CLOSE_A2_ACTION/trace.npz')
    same={k:bool(np.array_equal(a[k],b[k][:len(a[k])])) for k in ['sim_time','qpos','qvel','ctrl','contact_force_N']}
    videos=[];commands=[]
    for name in ['OPEN_CLOSE_A1_ACTION','OPEN_CLOSE_A2_ACTION']:
        case=BASE/name
        cc=json.loads((case/'commands.json').read_text(encoding='utf8'))
        commands.append(dict(case=name,count=len(cc),all_applied=all(c['applied'] for c in cc)))
        for video in ['replay_1x.mp4','comparison_baseline_1x.mp4']:
            path=case/video;capture=cv2.VideoCapture(str(path))
            fps=float(capture.get(cv2.CAP_PROP_FPS));frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok0,img=capture.read();capture.set(cv2.CAP_PROP_POS_FRAMES,frames-1);ok1,last=capture.read();capture.release()
            videos.append(dict(path=path.relative_to(BASE).as_posix(),fps=fps,frames=frames,
                duration_s=frames/fps,first_frame_decoded=bool(ok0),last_frame_decoded=bool(ok1)))
    report=BASE/'result_report.md'
    refs=re.findall(r'\[[^\]]+\]\(([^)]+)\)',report.read_text(encoding='utf8'))
    missing=[r for r in refs if not (report.parent/r).exists()]
    if missing or not all(v['fps']==20 and v['first_frame_decoded'] and v['last_frame_decoded'] for v in videos):
        raise RuntimeError('Delivery missing link or unreadable video')
    sources=BASE/'analysis_sources';sources.mkdir(exist_ok=True)
    for name in ['target_open_close_geometry.py','target_open_close_review.py','target_open_close_interface.py',
                 'target_open_close_diagnostics.py','target_open_close_finalize.py']:
        shutil.copy2(ROOT/'validation'/name,sources/name)
    write(BASE/'delivery_check.json',dict(physics_steps_added=0,historical_15_rows_unchanged=True,
        new_rows=2,total_rows=17,local_rows_match_total_append=True,A1_A2_shared27s_actual_prefix=same,
        commands=commands,videos=videos,report_link_count=len(refs),missing_links=missing,
        no_SHA=True,scope='Only new delivery, history count and saved shared prefix; no full-repository audit'))
    handoff=ROOT/'\u4ea4\u63a5\u8bb0\u5f55.md';text=handoff.read_text(encoding='utf8')
    text=text.replace('### \u66f2\u7ebf\u5f15\u5bfc\u4e0e\u987a\u5e94\u53cd\u9988\u8de8\u7ba1\uff1a\u4fdd\u6301\u6210\u7acb\uff0c\u7b2c\u4e8c\u8f6e\u6709\u9650\u6355\u83b7\u4ecd\u672a\u7a33\u5b9a\u4ea4\u63a5\u5e76\u505c\u6b62',
        '### \u66f2\u7ebf\u5f15\u5bfc\u4e0e\u987a\u5e94\u53cd\u9988\u8de8\u7ba1\uff1a\u524d\u90e8\u5f00\u5408\u5b9e\u9645\u6267\u884c\uff0c\u6536\u62e2\u540e\u6293\u7ba1\u5931\u8d25\u5e76\u505c\u6b62',1)
    start=text.index('- **\u8303\u56f4\u4e0e\u6700\u65b0\u7ed3\u8bba\uff1a** \u6c34\u5e73\u4e24\u7ba1L\u5f62\u300120\u5173\u8282MuJoCo\uff0c\u5b9e\u673a\u7531\u7528\u6237\u5b89\u6392\u3002')
    end=text.index('\n',start)
    replacement='- **\u8303\u56f4\u4e0e\u6700\u65b0\u7ed3\u8bba\uff1a** \u6c34\u5e73\u4e24\u7ba1L\u5f62\u300120\u5173\u8282MuJoCo\uff0c\u5b9e\u673a\u7531\u7528\u6237\u5b89\u6392\u3002\u9996\u6bb5\u7eed\u884cR2\u5b8c\u6574\u4fdd\u6301\u6210\u7acb\uff0c\u7b2c\u4e8c\u8f6e\u6709\u9650\u6355\u83b7\u4e09\u4e2aB\u5747\u5931\u8d25\u3001\u7d2f\u8ba115\u6b21\u3002\u7528\u6237\u968f\u540e\u7ea0\u504f\u8981\u6c42\u524d\u90e8\u5148\u5f20\u5f00\u518d\u6536\u62e2\uff1a\u672c\u8f6e\u65b0\u589e2/4\u6b21\u3001\u7d2f\u8ba117/19\u6b21\uff0c95\u2192130\u219295mm\u5f00\u5408\u5b9e\u9645\u6267\u884c\uff0c\u4f46\u6536\u62e2\u540e\u76ee\u6807\u63a5\u89e60\u20131\u3001\u7ba1\u8f74\u4e0d\u5728\u524d\u90e8\u6295\u5f71\u533a\u57df\uff0c84\u79d2\u5931\u8d25\u5e76\u505c\u6b62\u3002\u672a\u75282\u6b21\uff0c\u4e0d\u7ee7\u7eed\u539f\u5931\u7a33\u540e\u6bb5\uff0c\u4e0d\u80fd\u58f0\u79f0\u6539\u5584\u65e7\u5931\u8d25\u3002\u539f\u7b2c\u4e8c\u8f6eSTOPPED\u3001V1/V2/V3\u548c\u65e7\u5168\u8def\u5f84NOT_READY\u4fdd\u7559\uff0c\u51bb\u7ed3/F/\u5b9e\u673a0\uff1b\u4e0d\u6cdb\u5316\u4e3a\u6240\u6709\u66f2\u7ebf/\u987a\u5e94/\u5f00\u5408\u65b9\u6cd5\u65e0\u6548\u3002'
    text=text[:start]+replacement+text[end:]
    marker='### \u4ed3\u5e93\u76ee\u5f55\u6574\u7406\uff1a\u6b65\u6001\u5165\u53e3\u3001\u516c\u5171\u6a21\u5757\u4e0e\u65b9\u6848\u6587\u6863\u5206\u79bb'
    addition='''- **\u7528\u6237\u6307\u5b9a\u524d\u90e8\u5f00\u5408\uff1a** \u65b0`target_open_close_reference.py`\u4ee5\u76ee\u6807\u6bb5\u5355\u4f4d\u5f27\u957fR95\u2192130\u219295\u3001\u4e24\u8fde\u6746\u6e10\u5165\uff0c\u4fdd\u6301\u6e90\u4fa7\u53c2\u8003\u548c\u5b8c\u6574\u8d77\u59cbR2\u5206\u652f\uff1b`target_open_close_control.py`\u6682\u505c\u5b8c\u5168\u5f20\u5f00\u65f6\u524d\u90e8A\u2013D\u3001\u6536\u62e2\u6062\u590d\uff0c\u540e\u90e8\u539f\u53cd\u9988\u4fdd\u7559\u3002984\u59ff\u6001/\u5168\u90e8\u547d\u4ee4\u907f\u78b0\u3001\u6700\u9ad86.587\u5ea6/\u79d2\uff1b\u4fee\u590d\u56fa\u5b9a\u5c3e\u90e8\u7ea6\u675f\u53ca\u6839/\u5173\u8282\u72ec\u7acb\u63d2\u503c\u9020\u6210\u7684\u5c3e\u90e8\u7a7f\u900f\uff0c\u65e7\u79bb\u7ebf\u5c1d\u8bd5\u4fdd\u7559\u3002\u539f\u80fd\u529b\u548c\u9650\u503c\u4e0d\u53d8\u3002
- **\u5f00\u5408\u5b9e\u9645\u7ed3\u679c\u53ca\u4ea4\u4ed8\uff1a** A1\u7b2c16\u6b21\u572827\u79d2\u56e0\u65e7\u6750\u6599\u505c\u6ede\u4e2d\u65ad\uff0c\u5c1a\u672a\u6536\u62e2\uff1bA2\u4ec5\u5c06\u65e7\u79fb\u9001\u505c\u6ede\u6539\u4e3a\u8bb0\u5f55\u5931\u8d25\uff0c\u5141\u8bb8\u6709\u9650\u5f00\u5408\u5b8c\u6210\uff0c\u5f62\u72b6/\u65f6\u5e8f/\u80fd\u529b\u76f8\u540c\uff0c84\u79d2\u6536\u62e2\u540e\u4fdd\u6301\u5931\u8d25\u3002\u5b9e\u9645\u81ea\u8eab6\u8282\u70b9\u7a7a\u95f4\u87ba\u65cb\u8fd1\u4f3cR\u7ea694.9\u2192128.3\uff0817\u79d2\uff09\u2192109.5\uff0874\u79d2\uff09\uff0c\u62df\u5408\u8f74\u672b\u6001\u7ea678\u5ea6\u504f\u79bb\u7ba1\u8f74\uff1b\u81ea\u8eabR\u3001\u8ddd\u7ba1\u8f74\u8ddd\u79bb\u3001\u8f74\u504f\u79fb\u5206\u522b\u8bb0\u5f55\uff0c\u77ed\u7a97\u8bef\u5dee\u4e0d\u80fd\u5ffd\u7565\u3002\u672b2\u79d2\u6f02\u79fb5.296mm\u867d\u4f4e\u4e8e6.45\u95e8\u69db\uff0c\u4f46\u76ee\u6807\u63a5\u89e60\u20131\u3001\u6295\u5f71\u533a\u57df\u4e0d\u542b\u7ba1\u8f74\uff0c\u4e0d\u80fd\u5199\u62b1\u7a33\u3002\u540e\u90e850ms\u5468\u671f\u672b\u81f3\u5c112\u63a5\u89e6\uff0c10ms\u8bb0\u5f55\u4e00\u6b211\u6a21\u57570.01\u79d2\uff1b\u65e0\u89e6\u5730/\u81ea\u78b0\u649e/\u5916\u529b/\u91cd\u7f6e\uff0c\u4e24\u6b212220\u6761\u547d\u4ee4\u5e94\u7528\u786e\u8ba4\u3002\u505c\u6b62\u5019\u9009\uff0c\u4e0d\u7b2c18/19\u6b21\u3001\u4e0d\u6269\u5c55\u5176\u4ed6\u65b9\u5411\u3002\u89c1[\u5f00\u5408\u7ed3\u679c](experiments/curve_guided_transfer_20261006/target_open_close_20261006/result_report.md)\u548c[A2\u5b9e\u9645\u6b63\u5e38\u901f\u5ea6\u5f00\u5408/\u6295\u5f71\u56de\u653e](experiments/curve_guided_transfer_20261006/target_open_close_20261006/OPEN_CLOSE_A2_ACTION/replay_1x.mp4)\uff0c\u540c\u76ee\u5f55\u542b\u65e7R2\u539f\u901f\u89c6\u89c9\u5bf9\u7167\u3001\u539f\u59cb\u8f68\u8ff9/\u63a5\u89e6/\u63a7\u5236/\u6e90\u7801\u548c\u524d15\u884c\u4e0d\u53d8\u7684\u53f0\u8d26\u3002\u6ca1\u6709\u8bc1\u660e\u534a\u5f84\u592a\u5c0f\u662f\u552f\u4e00\u539f\u56e0\u6216\u5168\u90e8\u5f00\u5408\u65b9\u6848\u65e0\u6548\u3002

'''
    if marker not in text:raise RuntimeError('Handoff marker missing')
    text=text.replace(marker,addition+marker,1);handoff.write_text(text,encoding='utf8')
    print(json.dumps(dict(new_dynamics=2,total_dynamics=17,shared_prefix=same,commands=sum(c['count'] for c in commands),
        videos=videos,report_links_valid=not missing),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
