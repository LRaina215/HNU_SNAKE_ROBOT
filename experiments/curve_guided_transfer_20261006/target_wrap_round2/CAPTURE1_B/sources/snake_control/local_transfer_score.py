"""Offline/diagnostic material scorer. Never a controller input.

Known L outlet plane y=75mm from elbow origin, same target contact filter as
the historical logger. Entire modules and actual target normal force are used.
"""
import numpy as np


class LocalTransferScore:
    modules = (8, 9)

    def __init__(self, initial_points, initial_forces):
        self.entries = {}
        self.confirm = {i: 0. for i in self.modules}
        self.entry_candidate = {i: None for i in self.modules}
        self.rows = []
        self.highwater = float(np.mean(np.asarray(initial_points)[[8,9,10],1]))
        self.last_progress_active_time = 0.
        if any(self.in_region(initial_points, initial_forces, i) for i in self.modules):
            raise ValueError('Scored adjacent material units must initially be outside target grasp region')

    @staticmethod
    def in_region(points, forces, module):
        nodes = np.asarray(points)[module:module+2]-np.array([1100.,0.,1000.])
        radial = np.linalg.norm(nodes[:,[0,2]],axis=1)
        return bool(np.all(nodes[:,1]>75.) and np.all((radial>=56.5)&(radial<120.)) and
                    np.asarray(forces)[1,module]>.1)

    def record(self, time_s, stage, feed_mm, points, forces, active_time_s, dt=.01):
        membership={i:self.in_region(points,forces,i) for i in self.modules}
        for i in self.modules:
            if membership[i] and self.entry_candidate[i] is None:
                self.entry_candidate[i]=dict(time_s=float(time_s),stage=stage,commanded_feed_mm=float(feed_mm),
                                            active_time_s=float(active_time_s))
            if not membership[i]:
                self.entry_candidate[i]=None
            self.confirm[i] = self.confirm[i]+dt if membership[i] else 0.
            if self.confirm[i]>=.5 and i not in self.entries:
                self.entries[i]=dict(self.entry_candidate[i],confirmed_time_s=float(time_s))
        actual_station=float(np.mean(np.asarray(points)[[8,9,10],1]))
        if actual_station>self.highwater+1.:
            self.highwater=actual_station;self.last_progress_active_time=float(active_time_s)
        stall = stage.startswith('feed_') and active_time_s-self.last_progress_active_time>=10.
        self.rows.append(dict(time_s=float(time_s),stage=stage,feed_mm=float(feed_mm),
            scored_modules=list(self.modules),actual_station_y_mm=actual_station,
            confirmed_entries={str(i):v for i,v in self.entries.items()},
            in_region=[membership[i] for i in self.modules],stall=bool(stall)))
        return bool(stall)

    def result(self):
        ordered=all(i in self.entries for i in self.modules)
        if ordered:
            a,b=[self.entries[i] for i in self.modules]
            ordered=bool(a['time_s']<b['time_s'] and a['stage'].startswith('feed_') and b['stage'].startswith('feed_'))
        final=self.rows[-1] if self.rows else {}
        return dict(modules=list(self.modules),entries=self.entries,ordered_active_entries=ordered,
            final_modules_in_target=bool(final and all(final['in_region'])),
            criterion='both actual module endpoints beyond y75 and radial 56.5..120mm, target force >.1N for .5s, enter in order during commanded feed',
            score_only=True,controller_feedback=False)
