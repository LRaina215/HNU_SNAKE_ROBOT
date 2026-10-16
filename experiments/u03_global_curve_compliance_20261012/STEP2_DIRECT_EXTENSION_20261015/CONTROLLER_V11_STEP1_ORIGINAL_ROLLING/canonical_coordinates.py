"""Fixed hardware-sigma bookkeeping for identical native joint integrals."""
import numpy as np
class CanonicalCoordinates:
    def __init__(self):
        self.material=np.arange(22)*64.5;self.bounds=np.linspace(0,self.material[-1],257);n=self.material;b=self.bounds
        self.A=np.maximum(0,np.minimum(n[2:,None],b[None,1:])-np.maximum(n[:-2,None],b[None,:-1]));self.modes=np.zeros((256,2,20))
        for component,mask in [(0,np.arange(20)%2==1),(1,np.arange(20)%2==0)]:
            ids=np.flatnonzero(mask);M=self.A[ids];self.modes[:,component,ids]=-M.T@np.linalg.solve(M@M.T,np.eye(len(ids)))
    def field(self,q):return np.tensordot(self.modes,q,axes=(2,0))
    def hinges(self,field):
        b=self.A@field;return -np.where(np.arange(20)%2,b[:,0],b[:,1])
