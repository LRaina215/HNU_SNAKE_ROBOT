"""Archive the one completed V5.1 rollout through the existing read-only analysis."""
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'analyze_v5_actual.py').read_text(encoding='utf-8')
s=s.replace("CONTROLLER_V5_SOURCE_CONTACT_DYNAMICS","CONTROLLER_V5_1_GRIP_PRELOAD_PROJECTION").replace('[2.1,3.,4.,5.]','[2.1,5.,10.,13.9]').replace('[0,20,30,40,50]','[0,30,70,110,139]').replace('total_actual_starts=32','total_actual_starts=33')
s=s.replace('V5 changed the failure mode: mean polar slip is smaller at5s, but posterior grip/shape targets relax, allocated normal load decreases, and source axial drift exceeds the original guard. This is not a performance improvement or unique causal proof.','V5.1 preserves more grip than V5, but source circumferential creep still exceeds the original given-pose guard at13.9s. No receiver contact or capture. This does not establish a unique cause or a completed Step2 action.')
exec(compile(s,str(P/'analyze_v5_actual.py'),'exec'),{'__file__':str(P/'analyze_v5_actual.py'),'__name__':'__main__'})
