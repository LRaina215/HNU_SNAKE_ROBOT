"""连续变径场景、接触几何及真实 viewer 快捷键/重置/转发验证。"""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/results/variable-pipe'
MODEL = ROOT / 'mujoco_model/Snake_Robot_o20_Variable_Pipe.xml'
PORT = 18793
sys.path[:0] = [str(ROOT/'codes'), str(ROOT/'visualizer')]


def geometry_checks():
    m=mujoco.MjModel.from_xml_path(str(MODEL)); d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)
    assert m.nu == 20 and m.nkey == 1 and d.ncon == 0
    r0,r1,z0,z1,height=m.numeric('pipe_profile').data
    def radius(z):
        t=np.clip((z-z0)/(z1-z0),0,1)
        return r0+(r1-r0)*(10*t**3-15*t**4+6*t**5)
    mask=np.array([0,1,0,0,0,0],np.uint8); gid=np.zeros(1,np.int32)
    errors=[]
    # 避开三角形共享边的浮点射线归属歧义，在面的内部采样。
    for z in np.linspace(.01,height-.01,130):
        for a in (.012,.033,.47,1.13):
            v=np.array([np.cos(a),np.sin(a),0.])
            distance=mujoco.mj_ray(m,d,.4*v+[0,0,z],-v,mask,1,-1,gid)
            assert distance>0
            errors.append(abs(.4-distance-radius(z)))
    assert max(errors)<.0001, max(errors)
    # 独立的球形接触探针，在等径/过渡区内侧 1mm 应接触，外侧 1mm 不接触。
    tree=ET.parse(MODEL); root=tree.getroot(); world=root.find('worldbody')
    world.remove(world.find("body[@name='head']"))
    for tag in ('actuator','sensor','keyframe'):
        root.remove(root.find(tag))
    for mesh in list(root.find('asset').findall('mesh')):
        if mesh.get('file'): root.find('asset').remove(mesh)
    probe=ET.SubElement(world,'body',name='probe',pos='0 0 1')
    ET.SubElement(probe,'freejoint'); ET.SubElement(probe,'geom',name='probe_ball',type='sphere',size='.005',mass='.01')
    pm=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode')); pd=mujoco.MjData(pm)
    max_contact_error=0.
    for z in (.3,.706,.78,.94,1.08,1.294,1.5):
        for gap in (-.001,.001):
            mujoco.mj_resetData(pm,pd); pd.qpos[:3]=[radius(z)+.005+gap,0,z];mujoco.mj_forward(pm,pd)
            contacts=[c for c in pd.contact if pm.geom('probe_ball').id in (c.geom1,c.geom2)]
            if gap<0:
                assert contacts,(z,gap)
                max_contact_error=max(max_contact_error,abs(min(c.dist for c in contacts)-gap))
            else: assert not contacts,(z,gap)
    assert max_contact_error<.0001,max_contact_error
    renderer=mujoco.Renderer(m,height=900,width=1200)
    cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.85];cam.distance=3.4;cam.azimuth=135;cam.elevation=-12
    renderer.update_scene(d,camera=cam)
    from PIL import Image
    Image.fromarray(renderer.render()).save(OUT/'scene.png');renderer.close()
    return dict(joints=m.nu,diameter_mm=[r0*2000,r1*2000],transition_z_m=[z0,z1],height_m=height,
                radial_surface_error_mm=max(errors)*1000,contact_probe_error_mm=max_contact_error*1000,initial_contacts=d.ncon)


def run_gait():
    import Adaptive_Pole_Climbing_Mujoco as gait
    gait.DEBUG=False
    gait.main()


def find_window(pid):
    user=ctypes.windll.user32; result=[]
    callback=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
    def visit(hwnd,_):
        owner=wintypes.DWORD();user.GetWindowThreadProcessId(hwnd,ctypes.byref(owner))
        text=ctypes.create_unicode_buffer(256);user.GetWindowTextW(hwnd,text,256)
        if owner.value==pid and 'MuJoCo' in text.value: result.append(hwnd)
        return True
    user.EnumWindows(callback(visit),0)
    return result[0] if result else None


def key(hwnd,letter,control=True,repeats=0):
    # 只投递给本测试子进程的窗口；不操作用户其他窗口或系统全局按键。
    user=ctypes.windll.user32
    user.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
    def post(code,up=False,repeat=False):
        scan=user.MapVirtualKeyW(code,0)
        param=1|(scan<<16)|((1<<24) if code in (0x25,0x26,0x27,0x28) else 0)|((3 if up else 1 if repeat else 0)<<30)
        assert user.PostMessageW(hwnd,0x101 if up else 0x100,code,param)
    if control: post(0x11)
    code = {"-": 0xBD, "=": 0xBB, "UP": 0x26, "DOWN": 0x28, "LEFT": 0x25, "RIGHT": 0x27}.get(letter)
    if code is None: code = ord(letter)  # Win32 OEM punctuation keys
    post(code)
    for _ in range(repeats): post(code,repeat=True)
    post(code,up=True)
    if control: post(0x11,up=True)


def main():
    from python_client import SnakeControlClient
    OUT.mkdir(parents=True,exist_ok=True)
    report={'geometry':geometry_checks()}
    with socket.socket() as probe: probe.bind(('127.0.0.1',PORT))
    appdir=OUT/'app';appdir.mkdir(exist_ok=True)
    exe=appdir/'SnakeRobotVisualizer.exe'
    shutil.copy2(ROOT/'visualizer/windows_app/bin/SnakeRobotVisualizer.exe',exe)
    shutil.copy2(ROOT/'validation/fixtures/app-state.json',appdir/'app-state.json')
    env=os.environ.copy();env.update(SNAKE_VISUALIZER_PORT=str(PORT),SNAKE_MUJOCO_HEADLESS='0',SNAKE_SINGLE_STEP='1',
        SNAKE_MUJOCO_MODEL=str(MODEL),SNAKE_CONTROL_VIA_VISUALIZER='1',SNAKE_VISUALIZER_ENABLED='1',
        SNAKE_GEOMETRY_EXCHANGE='1',SNAKE_MUJOCO_MAX_SECONDS='30',SNAKE_MUJOCO_TRACE=str(OUT/'controls.npz'),PYTHONIOENCODING='utf-8')
    for name in ('SNAKE_HELIX_RADIUS_MM','SNAKE_HELIX_PITCH_MM'):env.pop(name,None)
    app=child=None;client=SnakeControlClient(port=PORT)
    logpath=OUT/'controls.log'
    try:
        app=subprocess.Popen([str(exe)],cwd=appdir,env=env)
        with logpath.open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)),'--run'],env=env,
                stdin=subprocess.PIPE,stdout=log,stderr=log,text=True,cwd=OUT)
            deadline=time.monotonic()+15
            while True:
                try:
                    hwnd=find_window(child.pid);frame=client.get_geometry()
                    if hwnd and time.time()*1000-frame['sample_unix_ms']<500:break
                except (OSError,RuntimeError):pass
                assert child.poll() is None and time.monotonic()<deadline,logpath.read_text(encoding='utf-8')
                time.sleep(.1)
            def wait_for(text):
                until=time.monotonic()+4
                while text not in logpath.read_text(encoding='utf-8'):
                    assert child.poll() is None and time.monotonic()<until,logpath.read_text(encoding='utf-8')
                    time.sleep(.05)
            def send(command):
                child.stdin.write(command+'\n');child.stdin.flush();time.sleep(.2)
            send('gravity=0');key(hwnd,'U',repeats=5);wait_for('完成第 1 步')
            key(hwnd,'N');wait_for('完成第 2 步')
            key(hwnd,'J');wait_for('完成第 3 步')
            key(hwnd,'H');wait_for('完成第 4 步')
            key(hwnd,'K');time.sleep(.3)
            key(hwnd,'E');time.sleep(.2)
            for k in ('UP','DOWN','RIGHT','LEFT','-','=','-'):
                key(hwnd,k);time.sleep(.15)
            key(hwnd,'J');wait_for('完成第 5 步')
            key(hwnd,'R');wait_for('实验已重置并暂停');time.sleep(.5)
            key(hwnd,'U');time.sleep(1)
            key(hwnd,'G');time.sleep(.12);key(hwnd,' ',control=False);wait_for('已暂停；')
            pause_at=time.time()*1000;time.sleep(1)
            frame=client.get_geometry();seq=frame['sequence'];time.sleep(.3)
            assert client.get_geometry()['sequence']>seq
            assert not client.get_state()['torque_active']
            client.close()  # 随后等子进程结束超过服务端空闲超时，下一次查询新建连接。
            send('status');child.stdin.close();child.wait(timeout=40)
            assert child.returncode==0,logpath.read_text(encoding='utf-8')
        assert 'saved screenshot' not in logpath.read_text(encoding='utf-8')
        assert 'R=115mm' in logpath.read_text(encoding='utf-8') and 'P=190mm' in logpath.read_text(encoding='utf-8')
        data=np.load(OUT/'controls.npz');records=json.loads((OUT/'controls.npz.received.json').read_text())
        packets={(x['run_id'],x['delivery_id']):x for x in records};signs=np.where(np.arange(20)%2,-1,1)
        for ctrl,run,delivery in zip(data['ctrl'],data['applied_run_id'],data['applied_delivery_id']):
            if delivery:assert np.array_equal(ctrl,np.deg2rad(np.array(packets[(run,int(delivery))]['targets_deg'])*signs))
            else:assert not np.any(ctrl)
        before=data['reset_count']==0
        ends=np.flatnonzero(np.diff(np.r_[0,data['completed_steps']])==1)
        ends=ends[before[ends]]
        assert len(ends)==5,len(ends)
        assert np.allclose(data['roll_phase'][ends],[0,0,.15,0,.075]),data['roll_phase'][ends]
        assert np.array_equal(data['ctrl'][ends[0]],data['ctrl'][ends[1]])
        reset=np.flatnonzero(np.diff(np.r_[0,data['reset_count']])==1)[0]
        assert data['sim_time'][reset]==0 and np.array_equal(data['qpos'][reset],data['qpos'][0])
        mask=data['wall_unix_ms']>pause_at+100
        assert mask.sum()>30
        for name in ('qpos','ctrl','sim_time','roll_phase'):assert np.all(data[name][mask]==data[name][mask][0]),name
        assert not client.get_state()['torque_active']
        report['controls']=dict(native_viewer_keys=True,initial_actions=5,autorepeat_ignored=True,hold_phase_unchanged=True,
            reset_exact=True,paused_samples=int(mask.sum()),applied_packets=len(set(zip(data['applied_run_id'],data['applied_delivery_id'])))-1,
            adaptive_samples=int(data['compliance_active'].sum()),gravity_for_controls_test=0)
        # 用同一场景与转发链路做短程重力观察，不将不发散当作稳定攀爬。
        env.update(SNAKE_MUJOCO_HEADLESS='1',SNAKE_SINGLE_STEP='0',SNAKE_MUJOCO_MAX_SECONDS='6',
                   SNAKE_MUJOCO_TRACE=str(OUT/'gravity.npz'))
        with (OUT/'gravity.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)),'--run'],env=env,stdout=log,stderr=log)
            child.wait(timeout=20)
            assert child.returncode==0,(OUT/'gravity.log').read_text(encoding='utf-8')
        observed=np.load(OUT/'gravity.npz')
        assert np.all(np.isfinite(observed['qpos'])) and np.all(np.isfinite(observed['ctrl']))
        assert observed['completed_steps'][-1]>=2
        received=json.loads((OUT/'gravity.npz.received.json').read_text())
        packets={(x['run_id'],x['delivery_id']):x for x in received}
        for ctrl,run,delivery in zip(observed['ctrl'],observed['applied_run_id'],observed['applied_delivery_id']):
            if delivery: assert np.array_equal(ctrl,np.deg2rad(np.array(packets[(run,int(delivery))]['targets_deg'])*signs))
            else: assert not np.any(ctrl)
        report['gravity_observation']=dict(gravity_m_s2=9.81,sim_seconds=float(observed['sim_time'][-1]),
            mean_joint_height_start_m=float(observed['robot_height_m'][0]),
            mean_joint_height_end_m=float(observed['robot_height_m'][-1]),
            maximum_pipe_contacts=int(observed['pipe_contacts'].max()),
            completed_steps=int(observed['completed_steps'][-1]),
            adaptive_samples=int(observed['compliance_active'].sum()),stable_climbing_verified=False)
        report['scope']='Scene/contact and control workflow validation, not stable climbing or real hardware acceptance.'
        (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:
        client.close()
        for process in (child,app):
            if process is not None and process.poll() is None:
                process.terminate();process.wait(timeout=5)


if __name__=='__main__':
    run_gait() if '--run' in sys.argv else main()
