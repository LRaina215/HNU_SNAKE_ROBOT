"""Load-driven source unfolding on the same curve as active front winding.

Posterior source material retains its captured helix. A smooth, moving
material interval suppresses circumferential source tangent near the outlet,
increasing axial tangent while preserving local curve arc speed. Its outlet
and bridge start move together. Actual commands remain full curve integrals.
"""
import copy
import numpy as np
from snake_control.adaptive_front_winding_transfer import AdaptiveFrontWindingTransfer
from snake_control.unified_curve_transfer import LINK,smooth


class UnfoldingTransfer(AdaptiveFrontWindingTransfer):
    def __init__(self,model,data):
        super().__init__(model,data)
        self.entry_capture_enabled=False
        self.curve.target_entry_offsets[:]=0.
        self.curve.source_release=dict(amplitude=0.,length_m=1.5*LINK,requested_amplitude=0.,
                                      loaded_source_error_deg=0.,supply_deficit_mm=0.)

    def update_source_release(self,t,p,centers,q,support,error,locations,dt):
        spec=self.curve.planned_route_spec
        joint_locations=self.curve.reference_node_parameters[1:-1]
        source=joint_locations<spec['source_bound']
        loaded=source&(self.filtered_load>1.8)
        mismatch=float(np.degrees(error[loaded].max())) if loaded.any() else 0.
        deficit=max(0.,self.front_local_material-self.rear_local_material)
        strain=float(np.clip((mismatch-3.)/12.,0.,1.))
        supply=float(np.clip((deficit-.25*LINK)/(.75*LINK),0.,1.))
        demand=max(strain,.65*supply) if support[1].sum()>.1 else 0.
        previous=self.curve.source_release['amplitude']
        # Unfold to relieve measured demand; recover slowly only after actual
        # target handover, so a transient drop in torque does not reclose the
        # same source coil before it has fed into the bridge.
        desired=max(previous,demand) if not self.transferred_material_modules else demand
        amplitude=float(previous+np.clip(desired-previous,-.015*dt,.05*dt))
        extent=(1.5+1.5*smooth(amplitude))*LINK
        self.curve.source_release=dict(amplitude=amplitude,length_m=extent,requested_amplitude=desired,
            loaded_source_error_deg=mismatch,supply_deficit_mm=deficit*1000,
            posterior_boundary_curve_parameter_mm=(spec['source_bound']-extent)*1000,
            source_outlet_curve_parameter_mm=spec['source_bound']*1000,
            basis='source load and front/rear supply mismatch move a continuous unfolding interval; untouched posterior curve retains support')

    def command(self,data,t,forces,torque,support_forces=None):
        target,info=super().command(data,t,forces,torque,support_forces)
        spec=self.curve.planned_route_spec
        locations=self.curve.reference_node_parameters[1:-1]
        release=copy.deepcopy(self.curve.source_release)
        start=spec['source_bound']-release['length_m']
        release['nominal_joint_stations_in_release_interval']=(np.flatnonzero((locations>=start)&(locations<=spec['source_bound']))+1).tolist()
        release['nominal_material_station_coordinates_mm']=(locations*1000).tolist()
        info['source_unfolding']=release
        info['stage']='source_unfolding_feed_and_front_winding'
        info['adaptive_front_winding']['action']='load-driven source helix unfolds into bridge; posterior support stays helical; front keeps bounded winding; common window follows existing supported motion'
        info['adaptive_front_winding']['target_entry_capture_feedback']={}
        return target,info
