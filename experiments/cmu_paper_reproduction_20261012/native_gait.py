"""CMU direct joint waves on the unchanged native20 model, head first."""
import numpy as np
N=20;LM=.0645;DM=.058;DP=.113

def wave(A,ws,phase):
    n=np.arange(N)
    return A*np.sin(ws*n+phase+(n%2)*np.pi/2)

def window(A,ws,source_phase,t,f,nb,low=.4,m=5,scale=1):
    n=np.arange(N);delta=-np.sign(A*ws)*np.pi/2
    phase=source_phase-2*np.pi*f*t-delta
    nt=np.pi*DM*abs(f)/LM*scale*t
    q=np.where(n<=nt,wave(A,ws,phase),wave(A,ws,phase+delta))
    L=low+(1-low)/(1+np.exp(np.clip(-m*(nt-nb),-60,60)))
    q[(n<nb)&(n<=nt)]*=L
    return q,dict(nt=nt,L=float(L),phase=phase,delta=delta)

def spiral(A,ws,phase,ns,kr=10,kp=.75,m=30):
    p=LM/(((A/(2*np.sin(ws)))**2+1)*ws);r=A/(2*np.sin(ws))*p
    wsp=kp*p*LM/((kr*r)**2+(kp*p)**2);Ap=2*kr*r/(kp*p)*np.sin(wsp)
    n=np.arange(N);sig=lambda x:1/(1+np.exp(np.clip(-x,-60,60)))
    s=sig(m*(n-(ns-.5)))+sig(m*((ns+.5)-n))-1
    return wave(Ap,wsp,phase)*s+wave(A,ws,phase)*(1-s)
