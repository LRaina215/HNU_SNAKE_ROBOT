"""Current nominal window, local execution curve and actual material roles."""
from physics import mujoco,np


def draw_polyline(scene,points,color,radius=.002):
    for a,b in zip(points[:-1],points[1:]):
        if scene.ngeom>=scene.maxgeom:return
        geom=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([*color,1.]))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b);scene.ngeom+=1


def overlay(scene,m,info):
    roles=info.get('actual_material_role',np.zeros(21,int));head=m.body('head').id
    colors={0:[.55,.55,.55],1:[1.,.45,.05],2:[.20,.75,.35],3:[.5,.4,.8],4:[1.,.85,.05],5:[.5,.75,.5]}
    for i in range(scene.ngeom):
        g=scene.geoms[i]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
            body=int(m.geom_bodyid[g.objid])-head
            if 0<=body<21:g.rgba[:3]=colors[int(roles[body])]
    draw_polyline(scene,info.get('global_reference_points',info['reference_points']),[.1,.8,1.],.002)
    draw_polyline(scene,info['nominal_curve_dense'][::2],[1.,.2,.7],.002)
