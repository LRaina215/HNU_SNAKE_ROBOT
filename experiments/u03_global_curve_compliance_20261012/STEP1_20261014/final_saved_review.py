"""Read-only common-window and artifact review; no dynamics or hashes."""
from physics import *
import json
import cv2

comparison = json.loads((BASE/'COMPARISON.json').read_text())
out = []
keys = ['step1_runner.py', 'feedback_curve.py', 'material_observer.py', 'physics.py', 'visual_overlay.py', 'REGISTERED.json']
for row in comparison['results']:
    name = row['case']
    physical = BASE.parent/'A16' if name == 'N_OPEN_REUSE' else BASE/name
    case = BASE/name
    z = np.load(physical/'trace.npz')
    t = z['t']; selected = np.flatnonzero((t >= 150.-1e-6) & (t <= 234.9+1e-6))
    first, last = int(selected[0]), int(selected[-1])
    y = 1000*(z['centers'][last,:,1]-z['centers'][first,:,1])
    m = mujoco.MjModel.from_xml_path(str(physical/'model.xml')); d = mujoco.MjData(m)
    spin_indices = selected[::5].tolist()
    if spin_indices[-1] != last: spin_indices.append(last)
    rates = []
    for i in spin_indices:
        d.qpos[:] = z['qpos'][i]; d.qvel[:] = z['qvel'][i]; mujoco.mj_forward(m,d)
        tangent = z['points'][i,:-1]-z['points'][i,1:]
        tangent /= np.linalg.norm(tangent,axis=1)[:,None]
        rate = []
        for n in range(21):
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('head').id+n,v,0)
            rate.append(float(v[:3]@tangent[n]))
        rates.append(rate)
    spin = np.rad2deg(np.trapz(rates,x=t[spin_indices],axis=0))
    capture = cv2.VideoCapture(str(case/'actual_side_oblique_1x.mp4'))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = capture.get(cv2.CAP_PROP_FPS)
    video_read = []
    for i in [0, frames//2, frames-1]:
        capture.set(cv2.CAP_PROP_POS_FRAMES,i); ok, im = capture.read(); video_read.append(bool(ok and im is not None))
    capture.release()
    check = dict(case=name, common_interval_s=[float(t[first]),float(t[last])], common_each_Y_mm=y.tolist(),
                 common_min_Y_mm=float(y.min()), common_mean_Y_mm=float(y.mean()),
                 common_each_spin_deg=spin.tolist(), common_min_abs_spin_deg=float(abs(spin).min()),
                 video_frames=frames, video_fps=fps, video_first_middle_last_read=video_read)
    if name != 'N_OPEN_REUSE':
        check['frozen_files_equal']={key:(case/key).read_bytes()==(BASE/'R1_READY_FROZEN'/key).read_bytes() for key in keys}
        check['A16_model_initial_equal']={key:(case/key).read_bytes()==(BASE.parent/'A16'/key).read_bytes() for key in ['model.xml','initial.npz']}
        check['governor_diagnostics']=dict(
            max_matched_lead_m=float(z['target_progress_lead_m'].max()),
            max_requested_reduction=float((1-z['shared_speed_requested_ratio']).max()),
            reason_codes=np.unique(z['speed_feedback_reason']).tolist(),
            max_lost_nominal_S=float(z['lost_nominal_S'].max()))
        check['registered_load_steps_correct']=row['actual_load_steps']==row['expected_load_steps']
    assert frames == 2350 and fps == 10. and all(video_read),name
    assert all(check.get('frozen_files_equal',{}).values()),name
    assert all(check.get('A16_model_initial_equal',{}).values()),name
    out.append(check)
    print(name, 'commonY', round(y.min(),3), round(y.mean(),3), 'spin', round(abs(spin).min(),3), flush=True)
report=dict(scope='Saved-state/common-window review only; original MuJoCo3.2.7; no simulation steps', common_window_s=[150.,234.9],
            artifact_checks_passed=True, results=out, visual_review='Executor inspected nominal, receiving-load and source-drag actual dual-view overlays; independent review recorded separately')
(BASE/'FINAL_REVIEW.json').write_text(json.dumps(report,indent=2))
