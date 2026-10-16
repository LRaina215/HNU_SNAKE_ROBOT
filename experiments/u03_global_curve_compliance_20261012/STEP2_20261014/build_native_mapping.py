"""Copy only exercised mapping/source utilities, not known-target planners."""
from pathlib import Path
import ast,textwrap
base=Path(__file__).resolve().parent; old=base.parent/'STEP1_20261014'
parts=['"""Original STEP1 mapping/source functions; target geometry is supplied separately."""\nimport numpy as np\nLINK=.0645\n\nclass NativeMapping:\n']
for file,klass,names in [('polar_window_reference.py','PolarWindowReference',['smooth','coords','polar_derivatives']),('curve_reference.py','GlobalCurveReference',['transport_normal']),('overlap_window_reference.py','OverlapWindowReference',['phase_cells','feed_distance','source','body','material_anchor']),('continuous_window_reference.py','ContinuousWindowReference',['joint_bending','command'])]:
    source=(old/file).read_text(encoding='utf-8'); lines=source.splitlines(keepends=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==klass)
    for name in names:
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        start=min([node.lineno]+[n.lineno for n in node.decorator_list])-1
        method=''.join(lines[start:node.end_lineno])
        if name=='phase_cells':
            method='    @staticmethod\n    def phase_cells(S):\n        x=np.clip((S-5)/4,0,1)\n        return min(S,5)+4*(x-2.5*x**4+3*x**5-x**6)\n'
        parts.append(method+'\n')
(base/'native_mapping.py').write_text(''.join(parts),encoding='utf-8')
# Source-mode and reconstruction remain in the original curvature integral chain.
source=(old/'feedback_curve.py').read_text(encoding='utf-8'); tree=ast.parse(source); lines=source.splitlines(keepends=True)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reconstruct')
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)); obs=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='observe_all')
(base/'source_mode.py').write_text('import numpy as np\nfrom physics import mujoco\n'+''.join(lines[fn.lineno-1:fn.end_lineno])+'\n\nclass SourceMode:\n'+''.join(lines[obs.lineno-1:obs.end_lineno])+'\n',encoding='utf-8')
print('Extracted native mapping and source curvature feedback without parent target constructors.')
source=(old/'physics.py').read_text(encoding='utf-8');lines=source.splitlines(keepends=True)
fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='snapshot')
(base/'evaluation.py').write_text('"""Scene truth ONLY for logging, original safety and offline acceptance."""\nfrom physics import *\n'+''.join(lines[fn.lineno-1:fn.end_lineno])+'\n',encoding='utf-8')

