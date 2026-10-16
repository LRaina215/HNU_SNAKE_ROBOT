from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'mujoco_model').is_dir() and (p/'snake_control').is_dir())
sys.path.insert(0,str(ROOT))
import mujoco,numpy as np
from snake_control.suspended_supply_transfer import nodes
MODEL=ROOT/'mujoco_model/Snake_Robot_o20_Pipe_elbow90_horizontal.xml'


INITIAL=BASE/"initial.npz"
