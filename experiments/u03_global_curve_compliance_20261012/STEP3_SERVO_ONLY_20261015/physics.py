"""Private physics assets; controller helpers never import the shared checkout."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import mujoco
BASE=Path(__file__).resolve().parent
MODEL=BASE/'assets/model.xml'

def nodes(d):
    return np.vstack([d.site('s_head').xpos,*[d.joint('joint'+str(i)).xanchor for i in range(1,21)],d.site('s_tail').xpos])
