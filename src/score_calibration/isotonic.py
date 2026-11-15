from fractions import Fraction
from bisect import bisect_right
import math

def fit(training):
    groups={}
    for r in training:
        w,y=groups.get(r['score'],(0,0));groups[r['score']]=(w+r['weight'],y+r['weight']*r['label'])
    blocks=[]
    for s,(w,y) in sorted(groups.items()):
        blocks.append({'lower':s,'upper':s,'weight':w,'positive_weight':y})
        while len(blocks)>1 and blocks[-2]['positive_weight']*blocks[-1]['weight']>blocks[-1]['positive_weight']*blocks[-2]['weight']:
            right=blocks.pop();left=blocks.pop();blocks.append({'lower':left['lower'],'upper':right['upper'],'weight':left['weight']+right['weight'],'positive_weight':left['positive_weight']+right['positive_weight']})
    return [dict(b,probability=str(Fraction(b['positive_weight'],b['weight']))) for b in blocks]

def score(blocks,evaluation):
    starts=[b['lower'] for b in blocks];predictions=[];brier=Fraction();loss=0.0;infinite=False;total=0
    for r in sorted(evaluation,key=lambda r:r['id']):
        b=blocks[max(0,bisect_right(starts,r['score'])-1)];p=Fraction(b['probability']);w=r['weight'];y=r['label'];total+=w;brier+=w*(p-y)**2;true_p=p if y else 1-p
        if not true_p:infinite=True
        else:loss-=w*math.log(float(true_p))
        predictions.append({'id':r['id'],'probability':str(p),'label':y,'weight':w})
    return {'predictions':predictions,'evaluation_weight':total,'weighted_brier':str(brier/total) if total else None,'weighted_log_loss':loss/total if total and not infinite else None,'infinite_log_loss':infinite}
