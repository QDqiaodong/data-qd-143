from .common import require,ident,number,rows
from .isotonic import fit,score

def observations(value):
    result=[];seen=set()
    for r in rows(value,20000):
        require(r,'id score label weight');rid=ident(r['id']);s=number(r['score'],0,1000000);label=number(r['label'],0,1);w=number(r['weight'],1,1000000)
        if rid in seen:raise ValueError('duplicate observation')
        seen.add(rid);result.append({'id':rid,'score':s,'label':label,'weight':w})
    return result,seen

def analyze(data):
    require(data,'training evaluation');training,tids=observations(data['training']);evaluation,eids=observations(data['evaluation'])
    if not training:raise ValueError('training required')
    if tids & eids:raise ValueError('training/evaluation identity overlap')
    blocks=fit(training);return {'blocks':blocks,**score(blocks,evaluation)}
