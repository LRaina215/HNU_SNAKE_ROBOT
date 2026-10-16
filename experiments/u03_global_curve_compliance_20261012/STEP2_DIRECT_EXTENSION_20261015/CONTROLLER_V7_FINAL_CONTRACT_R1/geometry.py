"""Cylinder hypothesis and target frame. No scene or target labels."""
import numpy as np

def normalize(x):return x/max(float(np.linalg.norm(x)),1e-12)

class Geometry:
    def __init__(self,entry,axis,radius):
        self.entry=np.asarray(entry,float).copy();self.axis=normalize(np.asarray(axis,float));self.radius=float(radius)
    def copy(self):return Geometry(self.entry,self.axis,self.radius)
    def frame(self):
        e=np.array([1.,0.,0.]);e=normalize(e-(e@self.axis)*self.axis)
        return e,np.cross(e,self.axis)
    def coordinates(self,points):
        offset=np.asarray(points)-self.entry; axial=offset@self.axis; radial=offset-axial[...,None]*self.axis
        return axial,radial,np.linalg.norm(radial,axis=-1)
    def as_dict(self):return dict(entry_m=self.entry.tolist(),axis=self.axis.tolist(),radius_m=self.radius)
    @classmethod
    def from_dict(cls,value):return cls(value['entry_m'],value['axis'],value['radius_m'])

