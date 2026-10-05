"""Keep a larger source coil until the target coil has sustained coverage.

An experimental coordination rule, not a force-closure certificate. It uses
the same supplied geometry/support observations as TargetFrameAcquisition;
the runner states whether they are known or encoder-derived. A-D compliance,
rolling, formation and actuator limits are retained without pose intervention.
"""
import numpy as np
from target_frame_acquisition import TargetFrameAcquisition


class ObservableSupportTransfer(TargetFrameAcquisition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.acquisition_observed_seconds=0.
        self.release_observable=False
        self.current_rear_reserve=8
        self.release_credit=0.

    def rear_release_permitted(self):
        return (self.handed_off or self.n-self.rear>self.current_rear_reserve) and self.release_credit>=1.-1e-9

    def recruit(self,points,history,status,front_delta,dt):
        # Waiting for front acquisition must not bank many rapid releases.
        # Use the same phase-to-link conversion as front formation, capped at
        # one pending link; preserve the existing 0.6 s permission filter.
        self.release_credit=min(1.,self.release_credit+abs(front_delta)*self.window.joints_per_rad*self.follow)
        previous=self.rear
        info=super().recruit(points,history,status,front_delta,dt)
        if self.rear>previous:self.release_credit=max(0.,self.release_credit-(self.rear-previous))
        info['release_phase_credit']=self.release_credit
        return info

    def compute(self,geometry,measured,phase,dt,contact_window,migrate=True):
        status=self.support(np.asarray(geometry['points_mm']),contact_window)
        ready=(status['actual_prefix']>=10 and status['front_wrap']>=1.
            and status['contact_span_deg']>=180.)
        self.acquisition_observed_seconds=self.acquisition_observed_seconds+dt if ready else 0.
        self.release_observable=self.release_observable or self.acquisition_observed_seconds>=.6-1e-9
        # Eight of twenty joints preserve most of an original coil. This is a
        # declared design prior for this robot, not a count of true contacts.
        self.current_rear_reserve=0 if self.release_observable else 8
        q,info=super().compute(geometry,measured,phase,dt,contact_window,migrate)
        info.update(observable_release=self.release_observable,
            observable_seconds=self.acquisition_observed_seconds,
            observation_rear_reserve=self.current_rear_reserve)
        return q,info

    def configuration(self):
        return dict(**super().configuration(),observability_coordination=dict(
            reserve_joints=8,minimum_prefix=10,minimum_turns=1.,
            minimum_contact_span_deg=180.,hold_seconds=.6,
            release_budget='at most one pending link, accumulated from actual front rolling and following',
            limitation='Coverage evidence only, not a bearing-margin certificate.'))
