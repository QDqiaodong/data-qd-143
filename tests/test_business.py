
import copy,json,unittest,tempfile,subprocess,sys,hashlib,random,itertools
from pathlib import Path
from fractions import Fraction
from score_calibration.model import analyze
ROOT=Path(__file__).resolve().parents[1]
def sample():return json.loads((ROOT/'fixtures/input.json').read_text())
class Business(unittest.TestCase):
    def bad(self,d):
        with self.assertRaises((ValueError,TypeError)):analyze(d)
    def test_unknown_fields(self):
        d=sample();d['unknown']=1;self.bad(d)

    def test_fit_and_score(self):
        r=analyze(sample());self.assertEqual([b['probability'] for b in r['blocks']],['0','1/2','1']);self.assertEqual(r['blocks'][1]['upper'],30);self.assertEqual(r['weighted_brier'],'1/8');self.assertAlmostEqual(r['weighted_log_loss'],__import__('math').log(2)/2)
    def test_equal_score_weighted(self):
        d=sample();d['training']=[{'id':'a','score':1,'label':1,'weight':3},{'id':'b','score':1,'label':0,'weight':1}];r=analyze(d);self.assertEqual(r['blocks'][0]['probability'],'3/4')
    def test_infinite_loss(self):
        d=sample();d['evaluation']=[{'id':'wrong','score':0,'label':1,'weight':1}];r=analyze(d);self.assertTrue(r['infinite_log_loss']);self.assertIsNone(r['weighted_log_loss']);self.assertEqual(r['weighted_brier'],'1')
    def test_empty_eval(self):
        d=sample();d['evaluation']=[];r=analyze(d);self.assertIsNone(r['weighted_brier']);self.assertFalse(r['infinite_log_loss'])
    def test_no_label_leakage(self):
        d=sample();base=analyze(d)['blocks']
        for r in d['evaluation']:r['label']=1-r['label']
        self.assertEqual(analyze(d)['blocks'],base)
    def test_identity_overlap(self):
        d=sample();d['evaluation'][0]['id']='a';self.bad(d)
    def test_order(self):
        d=sample();d['training'].reverse();d['evaluation'].reverse();self.assertEqual(analyze(d),analyze(sample()))
    def test_optimal_against_all_partitions(self):
        rng=random.Random(9)
        for _ in range(20):
            y=[rng.randrange(2) for i in range(5)];d={'training':[{'id':str(i),'score':i,'label':v,'weight':1} for i,v in enumerate(y)],'evaluation':[]};blocks=analyze(d)['blocks'];cost=sum((Fraction(b['probability'])-y[i])**2 for b in blocks for i in range(b['lower'],b['upper']+1));costs=[]
            for cuts in itertools.product([0,1],repeat=4):
                groups=[];start=0
                for i,cut in enumerate(cuts):
                    if cut:groups.append((start,i+1));start=i+1
                groups.append((start,5));means=[Fraction(sum(y[a:b]),b-a) for a,b in groups]
                if means==sorted(means):costs.append(sum((means[j]-y[i])**2 for j,(a,b) in enumerate(groups) for i in range(a,b)))
            self.assertEqual(cost,min(costs))

class Command(unittest.TestCase):
    def call(self,source,output):
        return subprocess.run([sys.executable,'-m','score_calibration.cli','--source',str(source),'--output',str(output)],capture_output=True,text=True)
    def test_cli_replay(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'out.json';src=ROOT/'fixtures/input.json';r=self.call(src,out);self.assertEqual(r.returncode,0,r.stderr);first=out.read_bytes();self.assertEqual(json.loads(first)['source_sha256'],hashlib.sha256(src.read_bytes()).hexdigest());self.assertEqual(self.call(src,out).returncode,0);self.assertEqual(first,out.read_bytes())
    def test_failure_preserves(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'bad.json';out=Path(td)/'out.json';out.write_text('prior')
            for body in ['{}','{"a":1,"a":2}','NaN']:
                src.write_text(body);self.assertEqual(self.call(src,out).returncode,2);self.assertEqual(out.read_text(),'prior')
    def test_same_path_and_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'in.json';src.write_bytes((ROOT/'fixtures/input.json').read_bytes());before=src.read_bytes();self.assertEqual(self.call(src,src).returncode,2);self.assertEqual(src.read_bytes(),before);out=Path(td)/'link.json';out.symlink_to(src);self.assertEqual(self.call(src,out).returncode,2);self.assertEqual(src.read_bytes(),before)
