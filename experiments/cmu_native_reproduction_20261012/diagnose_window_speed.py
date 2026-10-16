from native_run import *
for name in ['N01','N03','N05']:
    z=np.load(BASE/name/'trace.npz');ids=(z['t']>=2)&(z['t']<=50);c=z['centers'][ids]
    speed=np.polyfit(z['t'][ids],c[:,18,0],1)[0]
    axial_spacing=np.median(abs(np.diff(c[:,15:21,0],axis=1)))
    nominal=np.pi*DM*.02/LM
    print(name,'observed axial speed',speed,'median carrying axial spacing',axial_spacing,'axial-boundary rate',speed/axial_spacing,'nominal',nominal,'scale',speed/axial_spacing/nominal)
