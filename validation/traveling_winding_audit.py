"""Whole-crossing audit independent of formation labels and controller targets.

Keep the old 100.5-mm per-module region test as a separate diagnostic. It is
not synonymous with crossing: unsupported end links can extend radially while
the complete robot has left the junction and rolls on the target branch.
The full-body acceptance window must cover an entire commanded rolling cycle.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from three_segment_review import context
from segmented_pole_experiment import BASE, pipe_frame, write_json
from front_support_experiment import coverage


def evaluate(z,m):
    corner,source,target,radius=pipe_frame(m)
    time=z['time']-z['time'][0]
    points=z['points_mm']
    centers=(points[:,:-1]+points[:,1:])/2
    relative=centers-corner
    axial=relative@target
    dist=np.linalg.norm(np.cross(relative,target),axis=2)
    source_dist=np.linalg.norm(np.cross(relative,source),axis=2)
    winding,_=coverage(points,corner,target)
    force=z['normal_force']
    contact=force[:,1]>.1
    source_free=~np.any(force[:,0]>.1,axis=1)
    other_free=~np.any(force[:,2]>.1,axis=1)
    support_span=np.max(np.where(contact,axial,-1e9),axis=1)-np.min(np.where(contact,axial,1e9),axis=1)
    point_progress=(points-corner)@target
    all_clear=np.min(point_progress,axis=1)>radius+29.
    target_end=float(np.max((m.numeric('pipe_centerline').data.reshape(-1,4)[:,:3]*1000-corner)@target))
    within_pipe=np.max(point_progress,axis=1)<target_end
    assigned=np.all(dist+10<source_dist,axis=1)
    # Two spatially separated support regions, >1.5 turns, whole chain beyond
    # the original cylinder envelope. No source/unused branch/ground support.
    physical=all_clear&within_pipe&assigned&source_free&other_free&~z['ground']&(winding>=1.5)&(support_span>=64.5)
    # Retain the previous strict full-body radius-band criterion at entry and
    # at the end of the rolling cycle. During rolling, report excursions rather
    # than require every free end to have identical radius at every phase.
    strict=np.convolve(np.all(z['region'],axis=1).astype(int),np.ones(60,dtype=int),mode='full')[:len(time)]>=60
    samples=[]
    begin=0
    for end in range(1,len(time)):
        if not physical[end]:
            begin=end+1
            continue
        if begin>=len(time): break
        while begin<end and not strict[begin]: begin+=1
        if not strict[begin] or not strict[end]: continue
        if abs(z['phase'][end]-z['phase'][begin])<2*np.pi: continue
        front_fraction=float(np.any(contact[begin:end+1,:10],axis=1).mean())
        rear_fraction=float(np.any(contact[begin:end+1,10:],axis=1).mean())
        advance=(centers[end].mean(axis=0)-centers[begin].mean(axis=0))@target
        if min(front_fraction,rear_fraction)<.95 or advance<64.5: continue
        samples.append(dict(start_seconds=float(time[begin]),end_seconds=float(time[end]),
            phase_advance_rad=float(abs(z['phase'][end]-z['phase'][begin])),
            center_advance_mm=float(advance),minimum_wrap_turns=float(winding[begin:end+1].min()),
            minimum_axial_clearance_mm=float(np.min((points[begin:end+1]-corner)@target)-radius-29.),
            minimum_contact_span_mm=float(support_span[begin:end+1].min()),
            front_half_contact_fraction=front_fraction,rear_half_contact_fraction=rear_fraction,
            maximum_module_radius_mm=float(dist[begin:end+1].max()),
            all_module_radius_band_fraction=float(np.all(z['region'][begin:end+1],axis=1).mean()),
            mean_tracking_error_deg=float(abs(z['error_deg'][begin:end+1]).mean()),
            maximum_tracking_error_deg=float(abs(z['error_deg'][begin:end+1]).max())))
        break
    report=dict(acceptance='strict radius band 0.6s at entry/end; all body on finite target branch; target-only distributed support; >=1.5 turns; >=2pi rolling and >=one link advance',
        full_crossing_and_continued_rolling=bool(samples),accepted_window=samples[0] if samples else None,
        legacy_all_modules_in_radius_band_passed=bool(strict[-1] and np.all(source_free[-60:]) and not z['ground'].any()),
        maximum_target_region_modules=int(z['region'].sum(axis=1).max()),
        final_target_region_modules=int(z['region'][-1].sum()),
        minimum_all_point_target_progress_final_mm=float(np.min((points[-1]-corner)@target)),
        final_wrap_turns=float(winding[-1]),floor_contact=bool(z['ground'].any()),
        final_center_progress_mm=float((centers[-1].mean(axis=0)-centers[0].mean(axis=0))@target),
        final_maximum_radius_mm=float(dist[-1].max()),
        limitation='This independent auditor uses true axes and contacts for scoring only. Controller privileges are recorded separately in plan.json; a pass does not establish hardware performance.')
    return report


def audit(folder):
    result,z,m,d,meta=context(folder)
    report=evaluate(z,m)
    write_json(folder/'whole_crossing_audit.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args=parser.parse_args()
    print(json.dumps(audit(BASE/'traveling_winding'/args.name),ensure_ascii=False,indent=2))
