"""Recruit against the observed target coil, without absolute junction progress.

After the rear leaves, the old junction's position along the new pipe is not
observable from a single wrapped coil. This experiment replaces only that
coordinate-dependent bookkeeping. It retains A-D at both active ends, Yamada,
radial formation, supported-prefix admission, phase and rear reserve rules.
"""
import numpy as np
from snake_control.support_transfer_winding import RadialAcquisitionWinding


class TargetFrameAcquisition(RadialAcquisitionWinding):
    def __init__(self,*args,conserve_middle=False,distinguish_lines=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.conserve_middle=conserve_middle
        self.distinguish_lines=distinguish_lines

    def transfer_region(self,positions):
        radial=np.linalg.norm(np.cross(positions-self.corner,self.target),axis=1)
        region=radial<self.radius+self.options.shell_radius_mm+self.options.clearance_mm
        if self.distinguish_lines and not self.handed_off:
            # Proximity to the new axis alone confuses the old coil near the
            # intersection with acquired body. Compare radial distances to
            # both hypothesised lines, without an absolute junction-progress test.
            source_radial=np.linalg.norm(np.cross(positions-self.corner,self.source),axis=1)
            region&=radial+self.options.region_margin_mm<source_radial
        return region

    def release_geometry_ready(self,points,prefix):
        if self.conserve_middle:
            # Release one rear joint for each additional physically acquired
            # front module. Forming commands alone do not supply body length.
            acquired=max(0,prefix-self.initial_front)
            return self.rear<min(self.n,self.initial_rear+acquired)
        # The parent's two-link forming lead, middle-length budget, acquired
        # prefix, contact history and shrinking six-joint reserve still apply.
        # This counts body links rather than distance to an unseen intersection.
        return self.rear<self.n and self.rear-prefix<=self.middle_max

    def handoff_geometry_ready(self,points,history,status):
        centers=(points[:-1]+points[1:])/2
        radial=np.linalg.norm(np.cross(centers-self.corner,self.target),axis=1)
        whole_coil=bool(np.all(radial<self.radius+self.options.shell_radius_mm+self.options.clearance_mm)
            and status['front_wrap']>=1.5 and status['actual_prefix']>=self.n-1)
        samples=np.asarray(history)[-60:,1]>.1
        distributed=len(samples)>=60 and np.all(np.any(samples[:,:10],axis=1)&np.any(samples[:,10:],axis=1))
        source_clear=len(history)>=60 and not np.any(np.asarray(history)[-60:,0]>.1)
        return bool(whole_coil and distributed),bool(source_clear),whole_coil

    def configuration(self):
        config=super().configuration()
        config['target_frame_acquisition']=dict(region='ordered target-radius band; optional source-distance disambiguation; no absolute axial test',
            release='body-index gap with unchanged admission/reserve/support conditions',
            one_for_one_measured_release=self.conserve_middle,
            distinguish_source_and_target_lines=self.distinguish_lines,
            handoff='whole measured target-radius coil >=1.5 turns and both halves supported 0.6s',
            limitation='Geometry and support hypotheses can be wrong; independent physical full-crossing audit unchanged.')
        return config
