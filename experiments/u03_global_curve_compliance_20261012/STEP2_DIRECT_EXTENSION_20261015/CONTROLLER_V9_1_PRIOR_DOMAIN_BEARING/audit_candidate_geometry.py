"""Offline original-scene initial geometry/load facts; no integration."""
import argparse,json
from physics import *
from contact_sensor import read_sensor,known_source,source_geometry

def run(args):
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m);z=np.load(args.initial);mujoco.mj_setState(m,d,z['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
    reg=json.loads((BASE/'REGISTERED.json').read_text(encoding='utf-8'));sensor=read_sensor(m,d);p=nodes(d);centers=.5*(p[:-1]+p[1:]);h=m.body('head').id;depth=np.zeros(6);force=np.zeros(4)
    names=['pipe_straight_01','pipe_sharp_in_02','pipe_sharp_out_03','pipe_straight_04','floor']
    for k,ct in enumerate(d.contact):
        gs=[ct.geom1,ct.geom2];robot=[h<=m.geom_bodyid[g]<h+21 for g in gs]
        if all(robot):depth[4]=max(depth[4],-ct.dist);continue
        if sum(robot)!=1:continue
        name=m.geom(gs[0] if not robot[0] else gs[1]).name
        if name in names:
            j=names.index(name);j=5 if j==4 else j;depth[j]=max(depth[j],-ct.dist)
            if j<4:f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f);force[j]+=max(0.,f[0])
    source=[ct for ct in sensor['contacts'] if known_source(ct,reg['source_known'])];qa=m.jnt_qposadr[m.actuator_trnid[:,0]];rho=np.linalg.norm(centers[:,1:]-[0,1],axis=1)
    facts=dict(simulation_steps=0,independently_approved=False,initial_file=str(Path(args.initial).resolve()),original_model=str(MODEL),physical_enableflags=int(m.opt.enableflags),hinges=m.nu,rods=21,
        head_tip_m=p[0].tolist(),M0_center_m=centers[0].tolist(),head_tangent=((p[0]-p[1])/.0645).tolist(),source_center_radii_m=rho.tolist(),
        all21_known_source_geometry=bool(source_geometry(centers,reg['source_known']).all()),maximum_center_miter_sum_xy_m=float((centers[:,0]+centers[:,1]).max()),
        source_straight_rods=int((centers[:,0]<.95).sum()),source_inlet_rods=int((centers[:,0]>=.95).sum()),all21_before_receiver_axial_start=bool((centers[:,1]<.15).all()),
        initial_receiver_normal_N=float(force[2:].sum()),source_vertical_N=float(sum(ct['world_force_N'][2] for ct in source)),robot_weight_N=28.01736,
        all_contact_materials_source_consistent=bool(all(known_source(ct,reg['source_known']) for ct in sensor['contacts'])),source_contact_materials=sorted(set(ct['material'] for ct in source)),
        original_depths_mm=(depth*1000).tolist(),joint_peak_deg=float(abs(d.qpos[qa]).max()*180/np.pi),torque_peak_Nm=float(abs(d.actuator_force).max()),qacc_peak=float(abs(d.qacc).max()),root_qacc=d.qacc[:6].tolist(),
        release_hinges=8,released_material_length_m=8*.0645,retained_source_material_length_m=13*.0645,
        retained_M8toM20_winding_deg=float(np.ptp(np.unwrap(np.arctan2(centers[8:,2]-1,centers[8:,1])))*180/np.pi),
        caveat='Given loose-front source arrangement and static forward only; no receiving precontact, no formation dynamics or physical stability proof.')
    Path(args.output).write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--initial',required=True);ap.add_argument('--output',required=True);run(ap.parse_args())
