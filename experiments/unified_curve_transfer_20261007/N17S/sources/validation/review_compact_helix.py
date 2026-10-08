"""Compare actual compact-helix trials with N15, without replaying dynamics."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'validation')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_fixed_spatial_shift import review


def compact_review(case):
    review(case)
    commands=json.loads((case/'commands.json').read_text());result=json.loads((case/'result.json').read_text())
    e=json.loads((case/'fixed_spatial_evidence.json').read_text());initial=json.loads((case/'initial_curve.json').read_text())
    geometry=result['compact_helix'];boundary=initial['initial_spline_node_parameters_mm'][0]+geometry['future_transition_length_mm']
    counts=np.array([sum(np.asarray(c['material_locations_mm'])>=boundary) for c in commands])
    tc=np.array([r['t_s'] for r in commands]);indices=np.flatnonzero(counts)
    with np.load(case/'trace.npz') as z:
        t=z['sim_time']-z['sim_time'][0];span=z['actual_front_target_winding_span_rad'].copy();centres=z['centers_mm'].copy()
    report=dict(name=case.name,curve=geometry,preflight_path='../COMPACT_HELIX_20261009/preflight.json',
        actual_result={k:result.get(k) for k in ['outcome','complete','physical_seconds','final_shift_mm','admissions','ordered_target_handover_modules','final_target_modules','final_source_modules','error','save_errors']},
        actual_front_wrap_initial_rad=float(span[0]),actual_front_wrap_final_rad=float(span[-1]),
        actual_checkpoints=e['checkpoints'],axis_feedback=e['axis_wrap_feedback'],video=e['video_readback'],
        nominal_uniform_compact_extension=dict(first_module_centre_in_uniform_extension_t_s=float(tc[indices[0]]) if len(indices) else None,
            maximum_number_of_module_centres_in_uniform_extension=int(counts.max()),
            final_modules=[i for i,u in enumerate(commands[-1]['material_locations_mm']) if u>=boundary],
            warning='nominal reference centres, not proof of actual helix membership; inherited hook and pitch ramp precede this uniform extension'),
        facts_vs_inference='raw dynamics establish motion and contact; geometric compactness is an ideal target property, not successful feeding or wrapping')
    if commands[-1].get('skin_curve_feedback'):
        skin=[r['skin_curve_feedback'] for r in commands]
        report['original_shell_curve_feedback']=dict(
            maximum_full_future_goal_skin_penetration_mm=float(max(r['maximum_predicted_goal_pipe_skin_penetration_mm'] for r in skin)),
            final_full_future_goal_skin_penetration_mm=skin[-1]['maximum_predicted_goal_pipe_skin_penetration_mm'],
            actual_maximum_in_commands_mm=float(max(r['maximum_actual_pipe_skin_penetration_mm'] for r in skin)),
            final_geometric_only_FK_residual_mm=commands[-1]['axis_wrap']['geometric_only_FK_residual_after_rms_mm'],
            final=skin[-1],residual_scope='N17 axis_wrap main residual includes 21 weighted shell-depth terms; geometric_only metric excludes these and can be compared with N16')
        fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
        axs[0].plot(tc,[r['maximum_predicted_goal_pipe_skin_penetration_mm'] for r in skin],label='Registered full future target maximum')
        axs[0].plot(tc,[r['maximum_actual_pipe_skin_penetration_mm'] for r in skin],label='Observed actual maximum')
        axs[0].set_ylabel('Original shell-pipe depth / mm')
        for module in [8,9,10,11,12,13]:axs[1].plot(tc,[r['predicted_full_goal_pipe_skin_penetration_mm'][module] for r in skin],label=f'Future M{module}')
        axs[1].set_ylabel('mm');axs[1].set_xlabel('Actual elapsed time / s')
        for ax in axs:ax.legend();ax.grid(alpha=.2)
        fig.savefig(case/'original_shell_feedback_review.png',dpi=150);plt.close(fig)
    (case/'compact_helix_evidence.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    base=ROOT/'experiments/unified_curve_transfer_20261007/N15B'
    with np.load(base/'trace.npz') as z:
        oldt=z['sim_time']-z['sim_time'][0];oldspan=z['actual_front_target_winding_span_rad'].copy();oldc=z['centers_mm'].copy()
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    axs[0].plot(oldt,oldspan,label='N15B: 314.6 mm pitch');axs[0].plot(t,span,label=case.name+': 180 mm pitch')
    axs[0].axhline(span[0]*.65,color='gray',linestyle='--');axs[0].set_ylabel('Actual front winding span / rad')
    axs[1].plot(oldt,(oldc[:,:7,1]-oldc[0,:7,1]).mean(1),label='N15B actual front mean +Y')
    axs[1].plot(t,(centres[:,:7,1]-centres[0,:7,1]).mean(1),label=case.name+' actual front mean +Y')
    axs[1].set_ylabel('mm');axs[1].set_xlabel('Actual elapsed time / s')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.savefig(case/'compact_vs_N15_actual.png',dpi=150);plt.close(fig)
    print(json.dumps({'case':case.name,'nominal_compact':report['nominal_uniform_compact_extension'],'final_span':float(span[-1])}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',type=Path);args=p.parse_args()
    for case in args.cases:compact_review(case)
