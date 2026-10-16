"""Direct CMU CASE2025 equations; n=0 at head, radians and seconds."""
import numpy as np
N=16
LM=.06604
DM=.0508
DP=.0889

def helix(A, ws, phase):
    n=np.arange(N)
    return A*np.sin(ws*n+phase+(n%2)*np.pi/2)

def parameters(A,ws):
    p=LM/(((A/(2*np.sin(ws)))**2+1)*ws)
    return A/(2*np.sin(ws))*p,p

def window(A,ws,phase,nt,nb,low=.4,m=5):
    n=np.arange(N);delta=-np.sign(A*ws)*np.pi/2
    q=np.where(n<=nt,helix(A,ws,phase),helix(A,ws,phase+delta))
    L=low+(1-low)/(1+np.exp(np.clip(-m*(nt-nb),-60,60)))
    q[(n<nb)&(n<=nt)]*=L
    return q,float(L)

def spiral(A,ws,phase,ns,kr=10,kp=.75,m=30):
    r,p=parameters(A,ws)
    wsp=kp*p*LM/((kr*r)**2+(kp*p)**2)
    Ap=2*kr*r/(kp*p)*np.sin(wsp)
    n=np.arange(N)
    sig=lambda x:1/(1+np.exp(np.clip(-x,-60,60)))
    s=sig(m*(n-(ns-.5)))+sig(m*((ns+.5)-n))-1
    return helix(Ap,wsp,phase)*s+helix(A,ws,phase)*(1-s)

def window_rate(f):
    # Eq10 corrected by adjacent prose: one cycle travels pi*DM.
    return np.pi*DM*abs(f)/LM

def estimate_diameter(centers):
    c=centers-centers.mean(0)
    _,_,v=np.linalg.svd(c,full_matrices=False)
    return 2*np.mean(np.linalg.norm(c@v[1:].T,axis=1))-DM
