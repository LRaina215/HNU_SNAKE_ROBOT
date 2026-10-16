from native_run import *
m=mujoco.MjModel.from_xml_path(str(MODEL));A=1.2150592994538565;ws=.183889173151851
for nb,seam,phase in [(8,.9,2.617993877991494),(6,.95,2.70526034059121),(6,.95,2.7925268031909267)]:
 d,s=placement(m,A,ws,phase,nb,seam,0);p=nodes(d);t=(p[nb]-p[nb+1])/LM;polar=np.arctan2(t[1],t[2]);d,s=placement(m,A,ws,phase,nb,seam,polar);r,pairs=snapshot(m,d)
 print(nb,seam,phase,'polar',polar,'tangent',(r['points'][nb]-r['points'][nb+1])/LM,'depth',r['depth'],'head',r['points'][0]);print(pairs)
