"""Private source/rough-cylinder geometry, G construction and native-curvature IK.

No integration, actual root writes, truth receiver or saved truth path access.
G uses only immutable given points and an independent length-constrained guide.
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from math import comb
from physics import mujoco,MODEL,BASE,nodes
from geometry import normalize
from native_mapping import NativeMapping,LINK
from contact_sensor import known_source

def skew(v):return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])

def coarse_helix_radius(pipe_radius):
    # The cylindrical head has29mm radius. Place the midpoint of each64.5mm
    # helical chord at that surface offset, including0.3mm nominal touch.
    pitch=.18266863819422827
    def midpoint_gap(radius):
        rate=1/np.sqrt(radius**2+(pitch/(2*np.pi))**2);a=pitch*rate/(2*np.pi)
        span=brentq(lambda h:(a*h)**2+4*radius**2*np.sin(rate*h/2)**2-LINK**2,LINK,2*LINK)
        return radius*np.cos(rate*span/2)-(pipe_radius+.029-.0003)
    return brentq(midpoint_gap,pipe_radius+.029,pipe_radius+.05)

def coarse_model(geometry,source_plane=1.1):
    tree=ET.parse(MODEL);root=tree.getroot();asset=root.find('asset')
    allowed={'h_link','v_link','pipe_sharp_in_02'}
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') not in allowed:asset.remove(mesh)
        elif mesh.get('file'):mesh.set('file',str((MODEL.parent/mesh.get('file')).resolve()))
    world=root.find('worldbody')
    robot=world.find(".//body[@name='head']")
    robot_elements=set(robot.iter())
    for parent in world.iter():
        for element in list(parent):
            if element in robot_elements:continue
            if element.tag=='geom' and element.get('name') not in {'floor','pipe_straight_01','pipe_sharp_in_02'}:parent.remove(element)
            elif element.tag=='site':parent.remove(element)
    custom=root.find('custom')
    if custom is not None:root.remove(custom)
    g=geometry;e,b=g.frame();angles=np.linspace(0,2*np.pi,64,endpoint=False);rad=g.radius*(np.cos(angles)[:,None]*e+np.sin(angles)[:,None]*b)
    # Extrapolate the rough receiver to the KNOWN source cut plane. The slope,
    # radius and centerline remain the registered rough hypothesis throughout.
    cut=(source_plane-g.entry[0]-g.entry[1]-rad[:,0]-rad[:,1])/(g.axis[0]+g.axis[1])
    vertices=np.vstack([g.entry+cut[:,None]*g.axis+rad,g.entry+g.axis+rad])
    ET.SubElement(asset,'mesh',name='rough_receiver_prior',vertex=' '.join(map(str,vertices.ravel())))
    ET.SubElement(world,'geom',name='rough_receiver',type='mesh',mesh='rough_receiver_prior',rgba='.15 .55 .25 .2',contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    model.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
    return model

class CurvatureCoordinates:
    def __init__(self):
        self.material=np.arange(22)*LINK*1000;self.bounds=np.linspace(0,self.material[-1],257)
        n=self.material;b=self.bounds
        self.A=np.maximum(0,np.minimum(n[2:,None],b[None,1:])-np.maximum(n[:-2,None],b[None,:-1]))
        self.modes=np.zeros((256,2,20))
        for component,mask in [(0,np.arange(20)%2==1),(1,np.arange(20)%2==0)]:
            ids=np.flatnonzero(mask);M=self.A[ids];self.modes[:,component,ids]=-M.T@np.linalg.solve(M@M.T,np.eye(len(ids)))
    def field(self,q):return np.tensordot(self.modes,q,axes=(2,0))
    def hinges(self,field):
        b=self.A@field;return -np.where(np.arange(20)%2,b[:,0],b[:,1])

class RoughPlanner:
    def __init__(self,geometry,initial_qpos,initial_ctrl,settings):
        self.geometry=geometry.copy();self.p=settings;self.source_plane=settings.get('source_known',{}).get('miter_plane_sum_xy_m',1.1);self.m=coarse_model(geometry,self.source_plane);self.d=mujoco.MjData(self.m)
        m=self.m;self.head=m.body('head').id;self.qa=m.jnt_qposadr[m.actuator_trnid[:,0]];self.va=m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.initial_qpos=initial_qpos.copy();self.initial_ctrl=initial_ctrl.copy();self.initial_q=initial_qpos[self.qa].copy();self.observed_external_source_X_moment_Nm=0.
        self.coordinates=CurvatureCoordinates();self.q=self.initial_q.copy();self.command_local_offset=np.zeros(20);self.snapshot=None;self.anchor=settings['release_hinges']
        self.d.qpos[:]=initial_qpos;self.position();self.given=nodes(self.d).copy();self.initial_T=normalize(self.given[0]-self.given[1]);self.initial_head_R=self.d.xmat[self.head].reshape(3,3).copy()
        n=settings['wrap_hinges'];initialT=normalize(self.given[n-1]-self.given[n]);radial=normalize(np.r_[0.,self.given[n,1],self.given[n,2]-1]);axis=self.d.xaxis[m.actuator_trnid[n-1,0]]
        change=np.linspace(-np.radians(88)-self.initial_q[n-1],np.radians(88)-self.initial_q[n-1],501);directions=Rotation.from_rotvec(-change[:,None]*axis).apply(initialT)
        self.source_departure_T=directions[int(np.argmax(directions@radial-.1*abs(directions[:,0]-.3)))].copy()
        self.robot=[int(m.body_geomadr[self.head+i]) for i in range(21)]
        self.source=[m.geom(n).id for n in ['pipe_straight_01','pipe_sharp_in_02']];self.target=m.geom('rough_receiver').id
        # The actual3.2.7 collision pipeline can report greater source overlap
        # than NativeCCD distance. Query that SAME algorithm for known source
        # geometry too; the rough receiver is disabled in this private model.
        self.legacy_m=coarse_model(geometry,self.source_plane);self.legacy_m.opt.enableflags&=~64
        self.legacy_m.geom_contype[self.target]=0;self.legacy_m.geom_conaffinity[self.target]=0;self.legacy_d=mujoco.MjData(self.legacy_m)
        self.initial_legacy_gaps=self.legacy_source_contacts(21)[0].copy()
        self.initial_gaps={};self.contact_planes=[]
        calibration=np.load(BASE/'KNOWN_SOURCE_COIL_CALIBRATION.npz');self.coil_calibration=calibration['points'].copy();self.coil_integrals=calibration['joint_curvature_integrals'].copy();self.coil_virtual_radius=settings.get('source_virtual_radius_m',.0991869522236036);self.coil_source_radius=float(calibration['source_pipe_radius'])
        for i in range(21):
            for geom in self.source:self.initial_gaps[i,geom]=self.distance(self.robot[i],geom)[0]
        self.observe(initial_qpos,self.anchor);self.last_metrics={};self.last_guide=self.given.copy()
    def position(self):mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
    def legacy_source_contacts(self,n):
        lm=self.legacy_m;ld=self.legacy_d;ld.qpos[:]=self.d.qpos
        mujoco.mj_fwdPosition(lm,ld)
        gaps=np.full((n,2),.025);points=np.zeros((n,2,3));normals=np.zeros_like(points)
        for ct in ld.contact:
            gs=[int(ct.geom1),int(ct.geom2)];robot=[self.head<=lm.geom_bodyid[g]<self.head+21 for g in gs]
            if sum(robot)!=1:continue
            j=int(robot[1]);material=int(lm.geom_bodyid[gs[j]])-self.head;other=gs[1-j]
            if material>=n or other not in self.source:continue
            k=self.source.index(other)
            if ct.dist<gaps[material,k]:
                gaps[material,k]=ct.dist;points[material,k]=ct.pos;normals[material,k]=(1 if j else -1)*ct.frame[:3]
        return gaps,points,normals
    def set_geometry(self,g):
        self.geometry=g.copy();self.m=coarse_model(g,self.source_plane);self.d=mujoco.MjData(self.m)
    def observe(self,qpos,anchor):
        self.anchor=int(anchor);self.d.qpos[:]=qpos;self.position();body=self.head+self.anchor
        self.snapshot=dict(qpos=qpos.copy(),rootR=Rotation.from_quat(qpos[3:7][[1,2,3,0]]).as_matrix(),anchorP=self.d.xpos[body].copy(),anchorR=self.d.xmat[body].reshape(3,3).copy())
    def observe_contacts(self,contacts):
        # A local measured plane has no environment name or true geometry label.
        self.d.qpos[:]=self.snapshot['qpos'];self.position();self.contact_planes=[]
        origin=np.r_[0.,self.p['source_known']['center_yz_m']];self.observed_external_source_X_moment_Nm=sum(float(np.cross(ct['point']-.5*ct['penetration_m']*ct['normal']-origin,ct['world_force_N'])[0]) for ct in contacts if not known_source(ct,self.p['source_known']))
        for ct in contacts:
            i=ct['material'];R=self.d.xmat[self.head+i].reshape(3,3);local=R.T@(ct['point']-self.d.xpos[self.head+i])
            self.contact_planes.append(dict(material=i,local=local,point=ct['point'].copy(),normal=ct['normal'].copy(),depth=ct['penetration_m']))
    def fk(self,q):
        s=self.snapshot;self.d.qpos[:]=s['qpos'];self.d.qpos[self.qa]=q;self.position();body=self.head+self.anchor
        R=s['anchorR']@self.d.xmat[body].reshape(3,3).T;translation=s['anchorP']-R@self.d.xpos[body]
        self.d.qpos[:3]=R@s['qpos'][:3]+translation;self.d.qpos[3:7]=Rotation.from_matrix(R@s['rootR']).as_quat()[[3,0,1,2]];self.position()
        return nodes(self.d).copy()
    def command_step_metrics(self,previous,candidate):
        """Full-C motion and measured-plane compression; no truth geometry."""
        def state(q):
            points=self.fk(q);P=self.d.geom_xpos[self.robot].copy();R=self.d.geom_xmat[self.robot].reshape(21,3,3).copy()
            gaps=[]
            for ct in self.contact_planes:
                body=self.head+ct['material'];point=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local']
                gaps.append(-ct['depth']+ct['normal']@(point-ct['point']))
            source_gap=self.legacy_source_contacts(21)[0]
            return points,P,R,np.array(gaps),source_gap
        old,P0,R0,gap0,source0=state(previous);new,P1,R1,gap1,source1=state(candidate)
        cosine=np.clip((np.einsum('nij,nij->n',R0,R1)-1)/2,-1,1)
        bound=np.linalg.norm(P1-P0,axis=1)+self.m.geom_rbound[self.robot]*np.sqrt(2*(1-cosine))
        growth=float(np.maximum(0.,np.minimum(-.0008,gap0-.0001)-gap1).max()) if len(gap0) else 0.
        source_growth=float(np.maximum(0.,np.minimum(-.0025,source0-.0001)-source1).max())
        return dict(maximum_node_step_m=float(np.linalg.norm(new-old,axis=1).max()),maximum_geometry_step_m=float(bound.max()),measured_plane_excess_compression_m=growth,known_source_command_excess_compression_m=source_growth)
    def point_J(self,point,body,jpa,jra,n):
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(self.m,self.d,jp,jr,point,body)
        return jp[:,self.va[:n]]-jpa+skew(point-self.snapshot['anchorP'])@jra,jr[:,self.va[:n]]-jra
    def distance(self,a,b):
        if np.linalg.norm(self.d.geom_xpos[a]-self.d.geom_xpos[b])>self.m.geom_rbound[a]+self.m.geom_rbound[b]+.025:return .025,np.zeros(6)
        near=np.zeros(6)
        if self.m.geom_type[a]>self.m.geom_type[b]:gap=mujoco.mj_geomDistance(self.m,self.d,b,a,.025,near);near=np.r_[near[3:],near[:3]]
        else:gap=mujoco.mj_geomDistance(self.m,self.d,a,b,.025,near)
        return min(float(gap),.025),near
    def goals(self,alpha,beta,theta=0.):
        g=self.geometry;e,b=g.frame();phi=np.radians(-165.)+theta
        touch=min(.00025,self.p['nominal_touchdown_inward_offset_m']) if self.given[0,2]>.92 else self.p['nominal_touchdown_inward_offset_m']
        normal=np.cos(phi)*e+np.sin(phi)*b;tip=g.entry+(g.radius-touch)*normal
        final=normalize(.90*e+.435*b);times=np.array([0.,.45,.75,1.])
        low=self.given[0].copy();low[:]=g.entry-.10*g.axis;low[0]=self.given[0,0];low[2]=1.-(g.radius+.036+.005)
        near=tip-.025*g.axis-.025*b
        if self.given[0,1]>.04 and self.given[0,0]<g.entry[0]-.035 and self.given[0,2]<.99:
            # Extend below the receiver before turning into it. A direct rising
            # diagonal touches while the front is still coiled on the source.
            # This is a fixed coarse-prior release corridor, not active search.
            times=np.array([0.,.25,.65,.82,1.]);under=self.given[0].copy();under[2]=1.-(g.radius+.036+.018)
            low=g.entry-.015*g.axis;low[0]=self.given[0,0];low[2]=under[2]
            near=tip-.010*g.axis-.035*b;knots=np.array([self.given[0],under,low,near,tip]);start_derivative=np.array([0.,0.,-.15])
        elif self.given[0,2]>.92:
            # A prescribed front-only clearance corridor below the receiver prior.
            # The source stays at its given place; this contains no search feedback.
            times=np.array([0.,.30,.60,.80,1.]);under=self.given[0].copy();under[2]=1.-(g.radius+.036+.013)
            low[2]=under[2];near=tip-.025*g.axis-.040*b;knots=np.array([self.given[0],under,low,near,tip]);start_derivative=np.array([0.,0.,-.20])
        else:knots=np.array([self.given[0],low,near,tip]);start_derivative=.20*g.axis+.05*b
        path=CubicSpline(times,knots,axis=0,bc_type=((1,start_derivative),(1,.06*g.axis+.10*b)))
        point=path(alpha);T=normalize((1-alpha)*self.initial_T+alpha*final)
        if beta>0:
            sign=self.p['receiver_winding_sign'];phi+=sign*np.radians(240.)*beta;rad=np.cos(phi)*e+np.sin(phi)*b;tan=-np.sin(phi)*e+np.cos(phi)*b
            axial=.18266863819422827*np.radians(240.)*beta/(2*np.pi)
            radius=coarse_helix_radius(g.radius);helixT=normalize(.18266863819422827/(2*np.pi)*g.axis+sign*radius*tan)
            blend=NativeMapping.smooth(beta/.20);center=g.entry+axial*g.axis+radius*rad
            point=(1-blend)*(g.entry+(g.radius-self.p['nominal_touchdown_inward_offset_m'])*rad)+blend*center
            T=normalize((1-blend)*final+blend*helixT)
        return point,T
    def nominal_contact_phase(self,beta):return getattr(self,'wrap_phase_origin_rad',np.radians(-165.))-self.p['receiver_winding_sign']*2*np.pi*(getattr(self,'wrap_axial_origin_m',0.)+self.p['front_wrapping_axial_lead_m']*NativeMapping.smooth(beta/.2))/.18266863819422827
    def start_wrap(self,alpha,theta=0.):
        # Store the prescribed G at the contact event, never the actual body.
        self.wrap_start_guide,self.wrap_start_goal,self.wrap_start_T,_=self.guide(alpha,0.,theta)
        axial,radial,_=self.geometry.coordinates(self.wrap_start_goal);e,b=self.geometry.frame()
        self.wrap_phase_origin_rad=float(np.arctan2(radial@b,radial@e)-theta);self.wrap_axial_origin_m=float(axial)
        # C has its own previous commanded native realization. Preserve that
        # projection at the phase boundary; never put these FK points into G.
        self.wrap_start_C=self.fk(self.q).copy();self.wrap_start_C_T=normalize(self.wrap_start_C[0]-self.wrap_start_C[1])
    def guide(self,alpha,beta,theta=0.):
        # Independent geometric chain. Given setup is immutable; actual FK never enters G.
        if beta>0:return self.wrap_guide(beta,theta)
        n=self.p['release_hinges'] if beta<=0 else self.p['wrap_hinges'];target,T=self.goals(alpha,beta,theta);out=self.given.copy()
        shift=target-out[0];out[:n]+=np.linspace(1,0,n,endpoint=False)[:,None]*shift
        anchor=self.given[n].copy();out[0]=target;out[1]=target-LINK*T;fixed=1
        if beta>=.2:
            g=self.geometry;e,b=g.frame();radius=coarse_helix_radius(g.radius);pitch=.18266863819422827;rate=1/np.sqrt(radius**2+(pitch/(2*np.pi))**2);aa=pitch*rate/(2*np.pi)
            span=brentq(lambda h:(aa*h)**2+4*radius**2*np.sin(rate*h/2)**2-LINK**2,LINK,2*LINK)
            fixed=min(n-2,int(np.radians(240)*beta/(rate*span)))
            axial,rad,_=g.coordinates(target);phi=np.arctan2(rad@b,rad@e)
            for j in range(fixed+1):out[j]=g.entry+(axial-aa*span*j)*g.axis+radius*(np.cos(phi+rate*span*j)*e+np.sin(phi+rate*span*j)*b)
        for _ in range(300):
            for i in range(fixed+1,n+1):out[i]=out[i-1]+LINK*normalize(out[i]-out[i-1])
            out[n]=anchor
            for i in range(n-1,fixed,-1):out[i]=out[i+1]+LINK*normalize(out[i]-out[i+1])
            if beta>0:
                # The bridge guide must go around the known source, not through it.
                # These are geometric point/segment projections, independent of FK.
                for i in range(fixed+1,n):
                    center=.5*(out[i]+out[i+1]);rad=center[1:]-[0,1];rho=np.linalg.norm(rad)
                    if center[0]+center[1]<1.108 and rho<.098:
                        original=.5*(self.given[i]+self.given[i+1]);direction=normalize(np.r_[0.,rad]) if rho>.04 else normalize(np.r_[0.,original[1],original[2]-1])
                        displacement=(.098-rho)*direction;out[i]+=displacement
                        if i+1<n:out[i+1]+=displacement
            if fixed>1:
                if abs(np.linalg.norm(out[fixed+1]-out[fixed])-LINK)<1e-7:break
                continue
            out[1]=out[2]+LINK*normalize(out[1]-out[2]);out[0]=out[1]+LINK*T
            if np.linalg.norm(out[0]-target)<1e-7:break
            out[0]=target;out[1]=target-LINK*T
        if beta>0:
            # A smooth bridge replaces residual source coils. Endpoint-only FABRIK
            # otherwise retains the initial winding and can trap C at the source.
            m=n-fixed;A=anchor;B=out[fixed].copy();delta=B-A;D=np.linalg.norm(delta);u=normalize(delta)
            v=np.array([0.,1.,-1.]);v=normalize(v-v@u*u)
            if D<m*LINK-1e-9:
                omega=brentq(lambda angle:LINK*np.sin(angle/2)/np.sin(angle/(2*m))-D,1e-5,2*np.pi-1e-5)
                radius=LINK/(2*np.sin(omega/(2*m)));t=np.linspace(0,1,m+1);z=(t-.5)*omega
                arc=.5*(A+B)+radius*(np.sin(z)[:,None]*u+(np.cos(z)-np.cos(omega/2))[:,None]*v)
                blend=NativeMapping.smooth(beta/.4);out[fixed:n+1]=(1-blend)*out[fixed:n+1]+blend*arc[::-1]
                if blend>=1-1e-12:return out,target,T,n
        # Finish from fixed posterior anchor to preserve every material span exactly.
        out[n]=anchor
        for i in range(n-1,fixed-1,-1):out[i]=out[i+1]+LINK*normalize(out[i]-out[i+1])
        if fixed==1:out[0]=out[1]+LINK*normalize(target-out[1])
        else:
            for i in range(fixed-1,-1,-1):out[i]=out[i+1]+LINK*normalize(out[i]-out[i+1])
        return out,target,T,n
    def wrap_guide(self,beta,theta,native=False):
        """Fixed source endpoint, continuous smooth bridge, receiver helix.

        The bridge's free geometric amplitude is solved so that exactly n
        native-length chords end at the prescribed head. No FK enters this G.
        """
        n=self.p['wrap_hinges'];g=self.geometry;e,b=g.frame();pitch=.18266863819422827;phi0=getattr(self,'wrap_phase_origin_rad',np.radians(-165.))+theta
        touch=min(.00025,self.p['nominal_touchdown_inward_offset_m']) if self.given[0,2]>.92 else self.p['nominal_touchdown_inward_offset_m']
        virtual_radius=g.radius+.039  # Step1/A16 receiving virtual radius rule.
        blend=NativeMapping.smooth(beta/.2);radius=virtual_radius
        sign=self.p['receiver_winding_sign'];degrees=self.p['maximum_wrap_deg'];phi1=phi0+sign*np.radians(degrees)*beta;angle=np.linspace(phi0,phi1,max(2,int(degrees*beta)+2));axial=getattr(self,'wrap_axial_origin_m',0.)+pitch*sign*(angle-phi0)/(2*np.pi)+self.p['front_wrapping_axial_lead_m']*blend
        target=g.entry+axial[:,None]*g.axis+radius*(np.cos(angle)[:,None]*e+np.sin(angle)[:,None]*b);template_spline=None
        if native:
            # Model-aware C target. A smooth virtual G does not equal the nodes
            # of the alternating-hinge skeleton. Calibrate that stagger from an
            # immutable known-source pose, never from actual receiving-body FK.
            cp=self.coil_calibration;rho=np.linalg.norm(cp[:,1:]-[0,1],axis=1);rho=rho+g.radius-self.coil_source_radius
            axial_native=cp[:,0]-cp[0,0];phase=[phi1]
            for i in range(21):
                dx=axial_native[i+1]-axial_native[i];cosine=(rho[i]**2+rho[i+1]**2+dx*dx-LINK*LINK)/(2*rho[i]*rho[i+1]);phase.append(phase[-1]-sign*np.arccos(np.clip(cosine,-1,1)))
            phase=np.array(phase);axhead=axial[-1];points=g.entry+(axhead+axial_native)[:,None]*g.axis+rho[:,None]*(np.cos(phase)[:,None]*e+np.sin(phase)[:,None]*b)
            template_spline=CubicSpline(np.arange(22),points,axis=0);entry_parameter=float(np.interp(-sign*phi0,-sign*phase,np.arange(22)));target=template_spline(np.linspace(entry_parameter,0,max(2,int(degrees*beta)+2)))
        # Retain the unreleased part of the prescribed source coil. Creating one
        # long bridge from M14 immediately would uncoil six additional source
        # rods before a receiving wrap had stored their material length.
        material_per_turn=2*np.pi*np.sqrt(virtual_radius**2+(pitch/(2*np.pi))**2)/LINK
        receiving_material=float(entry_parameter) if native else degrees*beta/360.*material_per_turn
        cut=min(float(n),self.p['release_hinges']+receiving_material)
        source_spline=CubicSpline(np.arange(22),self.given,axis=0)
        source_parameters=np.linspace(float(n),cut,max(2,int((n-cut)*30)+2));source_points=source_spline(source_parameters)
        A=source_points[-1];B=target[0];sourceT=normalize(-source_spline(cut,1));tangent=-np.sin(phi0)*e+np.cos(phi0)*b
        entryT=normalize(-template_spline(entry_parameter,1)) if native else normalize((1-blend)*normalize(.90*e+.435*b)+blend*normalize(pitch/(2*np.pi)*g.axis+sign*radius*tangent))
        direction=normalize(np.array([0.,1.,-1.]));t=np.linspace(0,1,241);cache={}
        def build(height):
            cp=np.array([A,A+.08*sourceT,A+.16*sourceT+height*direction,B-.16*entryT+height*direction,B-.08*entryT,B])
            bridge=sum(comb(5,j)*(1-t[:,None])**(5-j)*t[:,None]**j*cp[j] for j in range(6));xyz=np.vstack([source_points[:-1],bridge,target[1:]])
            xyz=xyz[np.r_[True,np.linalg.norm(np.diff(xyz,axis=0),axis=1)>1e-10]]
            finalT=normalize(xyz[-1]-xyz[-2]);head_index=len(xyz)-1;xyz=np.vstack([xyz,xyz[-1]+np.linspace(.002,1.,200)[:,None]*finalT])
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))];spline=CubicSpline(arc,xyz,axis=0);headarc=arc[head_index]
            points=[self.given[n].copy()];loc=[0.]
            for _ in range(n):
                trials=loc[-1]+np.linspace(.1*LINK,3*LINK,80);distance=np.linalg.norm(spline(trials)-points[-1],axis=1);ids=np.flatnonzero(distance>=LINK)
                if not len(ids):raise ValueError('Bridge guide has no64.5mm forward chord')
                k=int(ids[0]);lo=loc[-1] if k==0 else trials[k-1];hi=trials[k]
                u=brentq(lambda u:np.linalg.norm(spline(u)-points[-1])-LINK,lo,hi,xtol=1e-11);loc.append(u);points.append(spline(u))
            cache.update(height=height,points=np.array(points),remainder=loc[-1]-headarc)
            return loc[-1]-headarc
        lo=build(0.);hi=build(.8)
        if lo*hi>0:height=0. if abs(lo)<abs(hi) else .8
        else:height=brentq(build,0.,.8,xtol=1e-9)
        build(height);out=self.given.copy();out[:n+1]=cache['points'][::-1];goal=target[-1]
        if native:direction=normalize(points[0]-points[1])
        else:
            h=pitch/(2*np.pi);span=brentq(lambda v:h*h*v*v+4*radius*radius*np.sin(v/2)**2-LINK*LINK,1e-6,np.pi)
            neck_phi=phi1-sign*span;neck=g.entry+(axial[-1]-h*span)*g.axis+radius*(np.cos(neck_phi)*e+np.sin(neck_phi)*b);direction=normalize(goal-neck)
        anchor=self.given[n].copy()
        if hasattr(self,'wrap_start_guide') and beta<.2:
            start=self.wrap_start_C if native else self.wrap_start_guide;startgoal=start[0] if native else self.wrap_start_goal;startT=self.wrap_start_C_T if native else self.wrap_start_T
            weight=NativeMapping.smooth(beta/.2);out=(1-weight)*start+weight*out
            goal=(1-weight)*startgoal+weight*goal
            direction=normalize((1-weight)*startT+weight*direction)
            anchor=(1-weight)*start[n]+weight*self.given[n]
        # A chord-root can fail to bracket for some bridge/helix combinations.
        # Close both geometric endpoints explicitly at every beta, preserving
        #64.5mm material spans. The virtual construction still has no FK input.
        for _ in range(1500):
            out[0]=goal;out[1]=goal-LINK*direction
            for i in range(2,n+1):out[i]=out[i-1]+LINK*normalize(out[i]-out[i-1])
            out[n]=anchor
            for i in range(n-1,0,-1):out[i]=out[i+1]+LINK*normalize(out[i]-out[i+1])
            out[0]=out[1]+LINK*direction
            if np.linalg.norm(out[0]-goal)<1e-8:break
        # Chord tangent is the realizable material tangent of G at its head.
        headT=normalize(out[0]-out[1]);return out,goal,headT,n
    def solve(self,alpha,beta=0.,theta=0.,max_iterations=None,position_cap=None,joint_step_rad=None):
        G,target,Tgoal,n=self.guide(alpha,beta,theta);geometric_targets=G.copy()
        if beta>0:geometric_targets,target,Tgoal,n=self.wrap_guide(beta,theta,True)
        self.last_geometric_targets=geometric_targets.copy();previous=self.q.copy();offset=self.command_local_offset.copy();previous_command_points=self.fk(previous+offset).copy();previous_command_planes=[]
        for ct in self.contact_planes:
            body=self.head+ct['material'];point=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local'];previous_command_planes.append(-ct['depth']+ct['normal']@(point-ct['point']))
        previous_points=self.fk(previous).copy();active=np.arange(n)
        # Each coordinate is a coefficient of the normalized curvature integral basis.
        cache={};source_pairs=[(self.robot[i],g,'source',i) for i in active for g in self.source]
        legacy_pairs=[(self.robot[i],g,'source_original_collision',i) for i in active for g in self.source]
        target_pairs=[(self.robot[i],self.target,'target',i) for i in active]
        self_pairs=[(self.robot[i],self.robot[j],'self',i) for i in active for j in range(i+2,21)]
        pairs=source_pairs+legacy_pairs+target_pairs+self_pairs
        def evaluate(qfront):
            if 'q' in cache and np.array_equal(qfront,cache['q']):return cache['r'],cache['J']
            q=previous.copy();q[:n]=qfront;field=self.coordinates.field(q);q=self.coordinates.hinges(field);points=self.fk(q)
            jpa=np.zeros((3,self.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(self.m,self.d,jpa,jra,self.snapshot['anchorP'],self.head+self.anchor);jpa=jpa[:,self.va[:n]];jra=jra[:,self.va[:n]]
            jp,jr=self.point_J(points[0],self.head,jpa,jra,n);T=self.d.xmat[self.head].reshape(3,3)[:,2]
            hs=self.p['head_scale_m'];tw=self.p['tangent_weight']
            residual=[(points[0]-target)/hs,tw*(T-Tgoal),.005*(qfront-previous[:n])];jac=[jp/hs,-tw*skew(T)@jr,.005*np.eye(n)]
            for i in range(1,n):
                Jp,_=self.point_J(points[i],self.head+i,jpa,jra,n);weight=self.p['wrap_body_position_weight_per_m'] if beta>0 else .02/LINK
                residual.append(weight*(points[i]-geometric_targets[i]));jac.append(weight*Jp)
            if beta>0:
                # Native coiling curvature mode with one free material gauge.
                # It prevents positional IK from replacing a wrap by a shallow,
                # steep-pitch arc. The mode is calibrated on known source only.
                wrap=self.p['maximum_wrap_deg']*beta;weights=NativeMapping.smooth((wrap-(2*np.arange(3)+1)*37.)/74.)
                if weights.max()>0:
                    h=.18266863819422827/(2*np.pi);rs=self.coil_virtual_radius;rt=rs+self.geometry.radius-self.coil_source_radius;scale=(rt/(rt*rt+h*h))/(rs/(rs*rs+h*h))
                    # [axis,e,b] has determinant-1. Reversing radial phase for
                    # positive receiver winding cancels that reflection, so its
                    # native hinge chirality stays equal to the known source.
                    target_z=scale*(self.p['receiver_winding_sign']*self.coil_integrals[:6:2]+1j*self.coil_integrals[1:6:2]);z=qfront[:6:2]+1j*qfront[1:6:2];c=np.sum(weights**2*z*np.conj(target_z));gamma=np.angle(c)
                    wanted=np.exp(1j*gamma)*target_z;phase_J=np.zeros(n)
                    if abs(c)>1e-12:
                        phase_J[:6:2]=np.imag(weights**2*np.conj(target_z)/c);phase_J[1:6:2]=np.imag(1j*weights**2*np.conj(target_z)/c)
                    Jz=-1j*wanted[:,None]*phase_J
                    for k in range(3):Jz[k,2*k]+=1.;Jz[k,2*k+1]+=1j
                    factor=self.p['native_coiling_mode_weight'];residual.extend([factor*weights*np.real(z-wanted),factor*weights*np.imag(z-wanted)])
                    jac.extend([factor*weights[:,None]*Jz.real,factor*weights[:,None]*Jz.imag])
            balance=self.p.get('source_axis_balance',{});balance_moment=0.;balance_gradient=np.zeros(n);balance_activation=0.
            if balance.get('enabled',False):
                origin=np.r_[0.,self.p['source_known']['center_yz_m']]
                for i in range(21):
                    mass=self.m.body_mass[self.head+i];force=mass*self.m.opt.gravity;point=self.d.xipos[self.head+i];balance_moment+=np.cross(point-origin,force)[0]
                    if i<n:
                        Jp,_=self.point_J(point,self.head+i,jpa,jra,n);balance_gradient+=force[2]*Jp[1]-force[1]*Jp[2]
                balance_moment+=self.observed_external_source_X_moment_Nm;balance_activation=NativeMapping.smooth(alpha/balance['activation_alpha'])
                factor=balance_activation/balance['moment_scale_Nm'];residual.append(np.array([factor*(balance_moment-balance['target_net_moment_Nm'])]));jac.append((factor*balance_gradient)[None,:])
            gaps=[];limits=[];gradients=[];legacy_gaps,legacy_points,legacy_normals=self.legacy_source_contacts(n)
            for a,b,kind,i in pairs:
                if kind=='source_original_collision':
                    k=self.source.index(b);gap=legacy_gaps[i,k];limit=min(-self.p.get('known_source_original_collision_depth_m',.0025),self.initial_legacy_gaps[i,k]-.00005)
                    grad=np.zeros(n)
                    if gap<.0249:
                        Ja,_=self.point_J(legacy_points[i,k],self.head+i,jpa,jra,n);grad=legacy_normals[i,k]@Ja
                    gaps.append(gap);limits.append(limit);gradients.append(grad);violation=min(0.,gap-limit)
                    residual.append(np.array([violation/.001]));jac.append((grad/.001)[None,:] if violation<0 else np.zeros((1,n)))
                    continue
                gap,near=self.distance(a,b)
                if kind=='source':limit=min(-self.p['fore_source_soft_penetration_m'] if beta>0 else -.0008 if i==0 else .0005,self.initial_gaps[i,b]-.00005)
                elif kind=='self':limit=.001
                elif beta>0 and i==0:
                    touch=min(.00025,self.p['head_target_terminal_depth_m']) if self.given[0,2]>.92 else self.p['head_target_terminal_depth_m']
                    limit=-((1-NativeMapping.smooth(beta/.2))*touch+NativeMapping.smooth(beta/.2)*.0008)
                else:
                    touch=min(.00025,self.p['head_target_terminal_depth_m']) if self.given[0,2]>.92 else self.p['head_target_terminal_depth_m']
                    limit=.002*(1-alpha**8)-touch*alpha**8 if beta==0 and i==0 else -.0008 if beta>0 else .0015
                grad=np.zeros(n)
                if gap<.0249 and abs(gap)>1e-9:
                    normal=(near[:3]-near[3:])/gap;Ja,_=self.point_J(near[:3],int(self.m.geom_bodyid[a]),jpa,jra,n)
                    Jb=self.point_J(near[3:],int(self.m.geom_bodyid[b]),jpa,jra,n)[0] if kind=='self' else np.zeros_like(Ja);grad=normal@(Ja-Jb)
                gaps.append(gap);limits.append(-self.p['rough_receiver_uncertainty_m'] if kind=='target' else limit);gradients.append(grad);violation=min(0.,gap-limit);rescale=self.p['rough_receiver_soft_scale_m'] if kind=='target' else .001
                residual.append(np.array([violation/rescale]));jac.append((grad/rescale)[None,:] if violation<0 else np.zeros((1,n)))
            for ct in self.contact_planes:
                i=ct['material']
                if i>=n:continue
                body=self.head+i;predicted=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local'];gap=-ct['depth']+ct['normal']@(predicted-ct['point'])
                # Registered0.1mm linearization allowance permits curved
                # surface sliding. This is separate from actual3mm safety.
                allowance=self.p.get('contact_step_allowance_m',.0001)
                allowed=min(.0025,max(.0008,ct['depth']+allowance)) if ct['depth']<=.0025 else max(.0025,ct['depth']-.0005)
                limit=-allowed
                Jp,_=self.point_J(predicted,body,jpa,jra,n);gradient=ct['normal']@Jp;gaps.append(gap);limits.append(limit);gradients.append(gradient)
                residual.append(np.array([min(0.,gap-limit)/.001]));jac.append((gradient/.001)[None,:] if gap<limit else np.zeros((1,n)))
            motion=np.empty(0);motion_J=np.empty((0,n))
            if position_cap is not None:
                displacement=points[:n]-previous_points[:n];Jpoints=np.array([self.point_J(points[i],self.head+i,jpa,jra,n)[0] for i in range(n)]);motion=1-np.sum(displacement**2,axis=1)/position_cap**2;motion_J=-2*np.einsum('ij,ijk->ik',displacement,Jpoints)/position_cap**2
            # Optimize against the SAME incremental compression rule used
            # by the final total-command filter. Existing preload can command
            # virtual overlap; its baseline is retained, never treated as new
            # physical penetration. All original geometric checks above stay.
            geometric_gap_count=len(gaps);constraint_labels=[('geometric',)+pair for pair in pairs]+[('geometric','measured_plane',ct['material']) for ct in self.contact_planes if ct['material']<n]
            command_points=self.fk(q+offset);jpa=np.zeros((3,self.m.nv));jra=np.zeros_like(jpa);mujoco.mj_jac(self.m,self.d,jpa,jra,self.snapshot['anchorP'],self.head+self.anchor);jpa=jpa[:,self.va[:n]];jra=jra[:,self.va[:n]]
            for index,ct in enumerate(self.contact_planes):
                i=ct['material']
                if i>=n:continue
                body=self.head+i;point=self.d.xpos[body]+self.d.xmat[body].reshape(3,3)@ct['local'];gap=-ct['depth']+ct['normal']@(point-ct['point']);limit=min(-.0008,previous_command_planes[index]-.0001);gradient=ct['normal']@self.point_J(point,body,jpa,jra,n)[0]
                gaps.append(gap);limits.append(limit);gradients.append(gradient);constraint_labels.append(('command','measured_plane',i));violation=min(0.,gap-limit)
                residual.append(np.array([violation/.001]));jac.append((gradient/.001)[None,:] if violation<0 else np.zeros((1,n)))
            if position_cap is not None:
                displacement=command_points[:n]-previous_command_points[:n];Jpoints=np.array([self.point_J(command_points[i],self.head+i,jpa,jra,n)[0] for i in range(n)])
                motion=np.r_[motion,1-np.sum(displacement**2,axis=1)/position_cap**2];motion_J=np.vstack([motion_J,-2*np.einsum('ij,ijk->ik',displacement,Jpoints)/position_cap**2])
            cache.update(constraint_labels=constraint_labels,geometric_gap_count=geometric_gap_count,source_axis_balance_moment_Nm=balance_moment,source_axis_balance_activation=balance_activation,q=qfront.copy(),r=np.concatenate(residual),J=np.vstack(jac),points=points,T=T,gaps=np.array(gaps),limits=np.array(limits),gradients=np.array(gradients),field=field,motion=motion,motion_J=motion_J)
            return cache['r'],cache['J']
        scale=np.radians(8.) if joint_step_rad is None else max(float(joint_step_rad),1e-9);lo=np.maximum(-1.,(-np.radians(88)-previous[:n])/scale);hi=np.minimum(1.,(np.radians(88)-previous[:n])/scale)
        def obj(u):r,J=evaluate(previous[:n]+scale*u);return .5*r@r
        def grad(u):r,J=evaluate(previous[:n]+scale*u);return scale*J.T@r
        def con(u,jac=False):evaluate(previous[:n]+scale*u);return np.vstack([scale*cache['gradients']/.001,scale*cache['motion_J']]) if jac else np.r_[(cache['gaps']-cache['limits'])/.001,cache['motion']]
        fit=minimize(obj,np.zeros(n),jac=grad,bounds=list(zip(lo,hi)),method='SLSQP',constraints=[dict(type='ineq',fun=con,jac=lambda u:con(u,True))],options=dict(maxiter=max_iterations or self.p['max_iterations'],ftol=1e-8,disp=False))
        candidate=previous.copy();candidate[:n]+=scale*fit.x
        evaluate(previous[:n]);initial_objective=float(.5*cache['r']@cache['r']);initial_violation=float(np.max(cache['limits']-cache['gaps']));step=candidate-previous;accepted=False;backtracks=0;restoration=None;solution_kind='hold'
        # An unfinished SLSQP iteration can exceed its geometric trust region.
        # Shorten only this candidate direction, then recheck nonlinear motion,
        # clearances and objective. Never accept a worsening feasible iterate.
        for backtracks in range(13):
            candidate=previous+step*(.5**backtracks);evaluate(candidate[:n]);violation=float(np.max(cache['limits']-cache['gaps']))
            accepted=bool(np.all(np.isfinite(candidate)) and violation<=.00005 and (position_cap is None or cache['motion'].min()>=-1e-7) and .5*cache['r']@cache['r']<=initial_objective+1e-8)
            if accepted:break
        if accepted:solution_kind='tracking'
        if not accepted and initial_violation>.00005:
            # Restore an already-infeasible sent C; retain original joint and
            # nonlinear motion/clearance acceptance rather than accepting a jump.
            def feasibility(u):
                con(u);v=np.minimum(0.,(cache['gaps']-cache['limits'])/.001);return .5*v@v
            def feasibility_gradient(u):
                con(u);v=np.minimum(0.,(cache['gaps']-cache['limits'])/.001);return scale*cache['gradients'].T@v/.001
            motion_constraints=[] if position_cap is None else [dict(type='ineq',fun=lambda u:(evaluate(previous[:n]+scale*u),cache['motion'])[1],jac=lambda u:(evaluate(previous[:n]+scale*u),scale*cache['motion_J'])[1])]
            restoration=minimize(feasibility,np.zeros(n),jac=feasibility_gradient,bounds=list(zip(lo,hi)),method='SLSQP',constraints=motion_constraints,options=dict(maxiter=max_iterations or self.p['max_iterations'],ftol=1e-8,disp=False))
            step=np.zeros(20);step[:n]=scale*restoration.x
            for backtracks in range(13):
                candidate=previous+step*(.5**backtracks);evaluate(candidate[:n]);violation=float(np.max(cache['limits']-cache['gaps']))
                accepted=bool(np.all(np.isfinite(candidate)) and violation<=.00005 and (position_cap is None or cache['motion'].min()>=-1e-7))
                if accepted:solution_kind='feasibility_restoration';break
        if accepted:self.q=candidate
        else:evaluate(previous[:n])
        self.last_guide=G.copy();self.last_metrics=dict(accepted=accepted,solution_kind=solution_kind,initial_clearance_violation_m=initial_violation,restoration_attempted=restoration is not None,restoration_solver_success=bool(restoration.success) if restoration is not None else False,restoration_iterations=int(restoration.nit) if restoration is not None else 0,solver_success=bool(fit.success),solver_iterations=int(fit.nit),clearance_violation_m=violation,
            candidate_backtracks=backtracks,objective_before=initial_objective,objective_after=float(.5*cache['r']@cache['r']),
            head_error_m=float(np.linalg.norm(cache['points'][0]-target)),source_axis_balance_moment_Nm=float(cache['source_axis_balance_moment_Nm']),source_axis_balance_activation=float(cache['source_axis_balance_activation']),G_head_goal_error_m=float(np.linalg.norm(G[0]-target)),
            G_material_length_error_m=float(abs(np.linalg.norm(np.diff(G,axis=0),axis=1)-LINK).max()),optimized_hinges=n,
            minimum_target_gap_m=float(min(cache['gaps'][k] for k,pair in enumerate(pairs) if pair[2]=='target')),
            geometric_clearance_violation_m=float(np.max(cache['limits'][:cache['geometric_gap_count']]-cache['gaps'][:cache['geometric_gap_count']])),command_incremental_clearance_violation_m=float(np.max(cache['limits'][cache['geometric_gap_count']:]-cache['gaps'][cache['geometric_gap_count']:])) if len(cache['gaps'])>cache['geometric_gap_count'] else 0.,command_incremental_plane_count=len(cache['gaps'])-cache['geometric_gap_count'],previous_local_offset_peak_rad=float(abs(offset).max()),most_active_clearance_pair=str(cache['constraint_labels'][int(np.argmax(cache['limits']-cache['gaps']))]))
        return self.q.copy(),G.copy(),self.last_metrics.copy()
    def gravity_torque(self,q,n):
        self.fk(q);out=np.zeros(20)
        for j in range(n):
            jid=self.m.actuator_trnid[j,0];pivot=self.d.xanchor[jid];axis=self.d.xaxis[jid];moment=np.zeros(3)
            for i in range(j+1):moment+=np.cross(self.d.xipos[self.head+i]-pivot,self.m.body_mass[self.head+i]*self.m.opt.gravity)
            # Fixed posterior bearing: increasing q rotates the free fore side negatively.
            # Holding torque is dU/dq = +axis dot the fore-side gravity moment.
            out[j]=axis@moment
        return out
