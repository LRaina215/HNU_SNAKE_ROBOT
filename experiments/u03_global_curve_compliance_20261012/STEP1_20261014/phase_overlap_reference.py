"""A14 mechanism contrast: only M decay S5..7 -> S5..9 versus A13."""
from overlap_window_reference import OverlapWindowReference
import numpy as np

class PhaseOverlapReference(OverlapWindowReference):
    @staticmethod
    def phase_cells(S):
        x=np.clip((S-5)/4,0,1)
        return min(S,5)+4*(x-2.5*x**4+3*x**5-x**6)

    def __init__(self,points,angles):
        super().__init__(points,angles)
        self.spec.update(version=14,mechanism_contrast='Only M slope decay extended from S5..7 to S5..9; final mu6->7. All A13 geometric construction, F, common psi, initialization and original physical driver unchanged.',source_spatial_phase='world Phi=beta*M, moving-source-chart x0 advancesa*M; M=.054*mu(S), mu slope decays5..9 to7',source_phase_decay_cells=[5,9],source_phase_final_mu=7,high_level_feedback=False)
