import random, time
from app.models import Station, Track, Train, SolveParams
from app.schedule import Network
from app.evaluation import Evaluator
from app.os_scheduler import os_schedule
from app.optimizer import optimize_order
from app.scenario import make_scenario
r=random.Random(3); N=21
st=[Station(id=f"S{i}",name=f"S{i}") for i in range(N)]
tk=[Track(id=f"S{i}-S{i+1}",a=f"S{i}",b=f"S{i+1}",single_track=(i%5==0)) for i in range(N-1)]
tr=[]
for k in range(120):
    a=r.randrange(0,N-4); b=r.randrange(a+3,N); route=[f"S{i}" for i in range(a,b+1)]
    if r.random()<.5: route.reverse()
    run=[r.uniform(3,9) for _ in range(len(route)-1)]
    tr.append(Train(id=f"T{k}",priority=r.randint(1,5),departure_min=r.uniform(0,600),route=route,run_min=run,dwells=[1.0]*(len(route)-1)))
net=Network(st,tk,tr)
p=SolveParams(disturb_count=10,iterations=600)
n=make_scenario(net,p); ev=Evaluator(n,p)
b=ev.evaluate(n.zero_holds()); h=os_schedule(n,p); o=ev.evaluate(h)
print("before",len(b.conflicts),round(b.fitness),"| OS",len(o.conflicts),round(o.fitness))
t=time.time(); bh,be,hist=optimize_order(n,ev,p)
print("optimizer",len(be.conflicts),round(be.fitness),"in",round(time.time()-t,1),"s; best curve",hist[0]["best"],"->",hist[-1]["best"])
