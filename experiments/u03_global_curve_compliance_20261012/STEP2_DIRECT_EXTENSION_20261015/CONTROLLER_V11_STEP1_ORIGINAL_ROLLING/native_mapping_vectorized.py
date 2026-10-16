"""Equivalent native Bishop integration using NumPy rotation prefix products.

Offline performance candidate; not selected by any existing frozen run.
No geometry, phase, material, servo or physics parameter changes.
"""
import numpy as np
from native_mapping import NativeMapping,LINK

def rotations(tangents):
    before=tangents[:-1];after=tangents[1:];cross=np.cross(before,after);sine=np.linalg.norm(cross,axis=1);cosine=np.clip(np.sum(before*after,axis=1),-1,1)
    axis=np.zeros_like(cross);mask=sine>1e-12;axis[mask]=cross[mask]/sine[mask,None]
    R=np.repeat(np.eye(3)[None,:,:],len(cross),axis=0);v=axis[mask];skew=np.zeros((len(v),3,3))
    skew[:,0,1]=-v[:,2];skew[:,0,2]=v[:,1];skew[:,1,0]=v[:,2];skew[:,1,2]=-v[:,0];skew[:,2,0]=-v[:,1];skew[:,2,1]=v[:,0]
    R[mask]=cosine[mask,None,None]*np.eye(3)+(1-cosine[mask,None,None])*v[:,:,None]*v[:,None,:]+sine[mask,None,None]*skew
    return R,axis,sine,cosine,mask

def prefix_products(R):
    product=R.copy();stride=1
    while stride<len(R):
        previous=product.copy();product[stride:]=previous[stride:]@previous[:-stride];stride*=2
    return product

def fast_transport_normal(normal,tangents):
    R,_,_,_,_=rotations(tangents)
    if len(R):normal=prefix_products(R)[-1]@normal
    normal-=normal@tangents[-1]*tangents[-1];return normal/np.linalg.norm(normal)

def integrate_native(tangents,arc,normal,head):
    R,axis,sine,cosine,mask=rotations(tangents);normals=np.vstack([normal,np.einsum('nij,j->ni',prefix_products(R),normal)])
    theta=np.arctan2(sine,cosine);span=np.diff(arc);before=tangents[:-1]
    derivative=np.cross(axis*(theta/span)[:,None],before);derivative[~mask]=0.
    field=np.column_stack([np.sum(derivative*normals[:-1],axis=1),np.sum(derivative*np.cross(before,normals[:-1]),axis=1)])/1000
    delta=span[:,None]*before
    delta[mask]=span[mask,None]*(np.sin(theta[mask,None])/theta[mask,None]*before[mask]+(1-np.cos(theta[mask,None]))/theta[mask,None]*np.cross(axis[mask],before[mask]))
    curve=head+np.vstack([np.zeros(3),np.cumsum(delta,axis=0)])
    return field,curve

class FastNativeMapping(NativeMapping):
    transport_normal=staticmethod(fast_transport_normal)
    def command(self,S):
        self.set_nominal_family(S);p,loc=self.body(S,True);L=loc[0]-loc[-1]
        arc=np.linspace(0,L,257);tt=-self.spline(loc[0]-arc,1);tt/=np.linalg.norm(tt,axis=1)[:,None]
        anchor,anchor_normal,angle,compensation=self.material_anchor(S,loc)
        path=np.linspace(anchor,loc[0],max(2,int((loc[0]-anchor)/.004)+1));srcT=-self.spline(path,1);srcT/=np.linalg.norm(srcT,axis=1)[:,None]
        normal=self.transport_normal(anchor_normal,srcT);chi=S/self.feed_per_cycle
        normal=normal*np.cos(angle)-np.cross(tt[0],normal)*np.sin(angle)
        initial_frame=np.column_stack([tt[0],normal,np.cross(tt[0],normal)])
        field,cc=integrate_native(tt,arc,normal,p[0]);bounds=arc*1000;material=(loc[0]-loc)*1000;b=self.joint_bending(bounds,field,material)
        q=-np.where(np.arange(20)%2,b[:,0],b[:,1]);C=np.column_stack([np.interp(material,bounds,cc[:,j]) for j in range(3)])
        canonical=np.eye(3)[np.argmin(abs(tt[0]))];canonical-=canonical@tt[0]*tt[0];canonical/=np.linalg.norm(canonical)
        rotation=np.column_stack([canonical,np.cross(tt[0],canonical)]).T@initial_frame[:,1:]
        return q,dict(S=S,chi=chi,roll_phase=angle,nominal_pipe_phase=np.zeros(2),local_material_phase=np.zeros(256),curvature_field=field,curvature_bounds_mm=bounds,material_arc_nodes_mm=material,material_rotation=rotation,reference_points=p,reference_locations=loc,reference_frame=initial_frame,nominal_curve_points=C,nominal_curve_dense=cc,source_reference=float(np.mean(loc<self.start_arc)),target_reference=float(np.mean(loc>self.target_arc)),connector_arc_m=self.connector_arc,nominal_shift_cells=S,source_station_transport_rad=compensation,uncoil_fraction=self.grow,receiving_recovery_fraction=self.recover,true_guide_body_arc_m=L,guide_to_C_max_mm=float(np.max(np.linalg.norm(C-p,axis=1)))*1000)
