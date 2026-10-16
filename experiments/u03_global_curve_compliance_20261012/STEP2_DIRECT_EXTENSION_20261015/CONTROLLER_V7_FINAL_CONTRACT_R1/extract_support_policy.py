"""Copy the preserved support/local-force policy without old phase integration."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parent
source=P.parents[1]/'STEP2_FROM_SOURCE_20261015/R6_FULL_BODY_SOURCE_GEOMETRY/source_controller.py'
text=source.read_text();lines=text.splitlines();tree=ast.parse(text);methods=[]
for node in tree.body:
    if isinstance(node,ast.ClassDef):
        for method in node.body:
            if isinstance(method,ast.FunctionDef) and method.name in ['support','adequate_supply','local_front']:methods.extend(lines[method.lineno-1:method.end_lineno])
out='"""Preserved source support and front-force policy; no phase advancement."""\nimport numpy as np\nfrom physics import mujoco\nfrom contact_sensor import known_source,source_geometry\n\nclass SourcePolicy:\n'+'\n'.join(methods)+'\n'
out=out.replace("geo=(abs(rho-(source['radius_m']+.036))<.031)&(c[:,0]>source['axial_bounds_m'][0])&(c[:,0]<source['axial_bounds_m'][1])","geo=source_geometry(c,source)")
(P/'support_policy.py').write_text(out)
