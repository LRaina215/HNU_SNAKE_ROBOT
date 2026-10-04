"""Run the vertical checkpoint only with the frozen horizontal selection."""
import argparse
import json

from segmented_pole_experiment import BASE, ROOT, VERTICAL, SegmentedOptions, run, sha


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['original','source_axis','segmented'],required=True)
    args=parser.parse_args()
    frozen=json.loads((BASE/'freeze.json').read_text(encoding='utf8'))
    assert sha(BASE/'plan.json')==frozen['plan_sha256'],'plan changed after freeze'
    for path,digest in frozen['source_hashes'].items():
        assert sha(ROOT/path)==digest, 'source changed after freeze: '+path
    for candidate in frozen['candidates']:
        assert sha(BASE/'horizontal'/f"final_segmented_w{candidate['width']}"/'result.json')==candidate['result_hash']
    result=run(VERTICAL,BASE/'vertical'/args.method,args.method,
               SegmentedOptions(transition_links=float(frozen['selected_width'])))
    print(json.dumps({k:result.get(k) for k in ['method','outcome','passed','steps','progress_mm','new_modules','error']}))
