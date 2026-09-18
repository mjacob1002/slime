#!/usr/bin/env bash
# True per-rollout TOTAL for either entry point.
#   train.py (colocate) prints THREE lines per rollout -- "Rollout N took" is GENERATION
#   ONLY (train.py:239), with "Training on rollout N took" (:265) and "Weight update N
#   took" (:298) separate. train_streaming.py:703 prints ONE line that already includes
#   training. Grepping both patterns into one column compares generation against total
#   and overstates colocate's speed by ~40%.
RID=$1
docker exec slime-dev-yi bash -c "grep -oE '(Rollout|Training on rollout|Weight update|Streaming rollout) [0-9]+ took [0-9.]+s' /root/shared_data/$RID/run.log 2>/dev/null" | python3 -c "
import sys,re,collections
d=collections.defaultdict(dict)
for l in sys.stdin:
    m=re.match(r'(Rollout|Training on rollout|Weight update|Streaming rollout) (\d+) took ([0-9.]+)s',l.strip())
    if m: d[int(m.group(2))][m.group(1)]=float(m.group(3))
for r in sorted(d):
    x=d[r]
    tot=x['Streaming rollout'] if 'Streaming rollout' in x else sum(x.values())
    print('r%d=%.0f'%(r,tot),end=' ')
print()
"
