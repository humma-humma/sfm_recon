"""Deterministic group quotas with shuffled coverage within each group."""
import random


def balanced_schedule(original,added,steps,seed=42):
    if steps<=0 or steps%10:raise ValueError('Steps must be a positive multiple of ten')
    if not original or not added:raise ValueError('Both groups must be nonempty')
    if len(set(original+added))!=len(original)+len(added):raise ValueError('Indices must be unique and disjoint')
    rng=random.Random(seed);pools=[[],[]];groups=[list(original),list(added)];schedule=[]
    for _ in range(steps//10):
        quota=[0]*9+[1];rng.shuffle(quota)
        for group in quota:
            if not pools[group]:
                pools[group]=groups[group].copy();rng.shuffle(pools[group])
            schedule.append(pools[group].pop())
    return schedule
