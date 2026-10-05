"""Known-axis, contact-assisted boundary migration for the simulation prototype.

Update only between closed front/rear phase cycles. This is a feasibility
controller using simulation contact truth, not a hardware obstacle estimator.
"""
import numpy as np


class SpatialBoundaryMigration:
    def __init__(self, initial=6., width=6., max_change=.25, persistence=3):
        self.value=initial;self.width=width;self.max_change=max_change
        self.persistence=persistence;self.candidate=None;self.streak=0

    def update(self, centers, corner, source, target, target_contacts,
               all_transferred=False):
        delta=np.asarray(centers)-corner
        ds=np.linalg.norm(delta-(delta@source)[:,None]*source,axis=1)
        dt=np.linalg.norm(delta-(delta@target)[:,None]*target,axis=1)
        dt+=np.maximum(0.,-(delta@target))
        cuts=np.arange(2,len(centers)+1)
        costs=[dt[:cut].sum()+ds[cut:].sum() for cut in cuts]
        candidate=float(cuts[np.argmin(costs)])
        near=int(candidate)
        corroborated=bool(np.any(target_contacts[max(0,near-3):min(len(centers),near+1)]))
        if all_transferred:
            # Move the whole transition beyond the tail; do not leave a
            # permanently source-directed tail after full physical transfer.
            candidate=len(centers)+self.width/2
            corroborated=bool(np.count_nonzero(target_contacts)>=2)
        if candidate==self.candidate and corroborated:self.streak+=1
        else:self.candidate=candidate;self.streak=int(corroborated)
        if self.streak>=self.persistence:
            self.value+=float(np.clip(candidate-self.value,-self.max_change,self.max_change))
        return self.value
