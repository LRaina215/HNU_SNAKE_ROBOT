from physics import mujoco,np
def draw_polyline(scene,points,color,radius=.002):
    for a,b in zip(points[:-1],points[1:]):
        if scene.ngeom>=scene.maxgeom:return
        g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([*color,1.]))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,radius,a,b);scene.ngeom+=1
def overlay(scene,m,info,estimate):
    roles=info['estimated_material_role'];h=m.body('head').id
    colors={0:[.55,.55,.55],1:[1.,.45,.05],2:[.2,.75,.35],3:[.5,.4,.8],4:[1.,.85,.05],5:[.5,.75,.5]}
    for i in range(scene.ngeom):
        g=scene.geoms[i]
        if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
            body=int(m.geom_bodyid[g.objid])-h
            if 0<=body<21:g.rgba[:3]=colors[int(roles[body])]
    if 'initial_global_family_dense' in info:draw_polyline(scene,info['initial_global_family_dense'],[.35,.45,.95],.001)
    draw_polyline(scene,info.get('global_reference_dense',info['global_reference_points']),[.1,.8,1.])
    draw_polyline(scene,info['nominal_curve_dense'][::2],[1.,.2,.7])
    draw_polyline(scene,np.array([estimate.entry-.05*estimate.axis,estimate.entry+.75*estimate.axis]),[1.,.85,.05],.0015)

