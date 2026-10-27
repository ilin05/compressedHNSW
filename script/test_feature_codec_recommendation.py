#!/usr/bin/env python3
"""Synthetic unit/integration fixtures only; these are NOT research results."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import feature_codec_recommendation as m


class RecommendationTests(unittest.TestCase):
    def test_bounded_deterministic_sampling(self):
        import random
        for n, count in ((1000000000,1024),(17,17),(1,1)):
            a=m.sample_ids(n,count,random.Random(100))
            self.assertEqual(a,m.sample_ids(n,count,random.Random(100)))
            self.assertEqual(len(set(a)),count)
            self.assertTrue(all(0 <= i < n for i in a))

    def test_xor_bit_width(self):
        vals=np.array([0,1,2,3,255,1<<63,(1<<64)-1],dtype=np.uint64)
        self.assertEqual(m.bit_length(vals).tolist(),[int(v).bit_length() for v in vals])

    def test_features_and_weight_switch(self):
        settings=dict(sample_pool_size=32,dimensions=8,references=4,dimension_scale=128.,codec_word_bits=64)
        zeros=np.zeros((32,8),np.float32)
        f=m.extract_features(zeros,zeros[:4],8,settings)
        self.assertEqual(f['zero_ratio'],1)
        self.assertEqual(f['integer_ratio'],1)
        self.assertEqual(f['decimal_complexity'],0)
        self.assertEqual(f['reference_equal_ratio'],1)
        self.assertEqual(f['reference_xor_width'],0)
        model=dict(pool='all',space_weight=.5,min_payload_ratio=1.02,size_normalizer=1,time_normalizer=1)
        p={'ALP':dict(size=.5,time=.1),'DeXOR':dict(size=.2,time=1),
           'Gorilla':dict(size=.7,time=1),'Elf':dict(size=.8,time=1)}
        self.assertEqual(m.select(model,p,1,0)[0],'DeXOR')
        self.assertEqual(m.select(model,p,0,1)[0],'ALP')
        self.assertEqual(m.select(model,{c:dict(size=1.2,time=1) for c in m.CODECS})[0],'raw')
        with self.assertRaises(ValueError): m.select(model,p,-1,1)

    def test_collect_tune_test_resume_and_standalone(self):
        # Artificial codecs/labels below exercise plumbing only. Never export
        # these fixture models as a calibrated model for real datasets.
        with tempfile.TemporaryDirectory(prefix='codec_feature_fixture_') as directory:
            root=Path(directory); data=root/'data';data.mkdir()
            alp=[]; differential=[]; names=[]
            rng=np.random.default_rng(97)
            for k in range(3):
                name='synthetic_fixture_'+str(k);names.append(name)
                x=rng.integers(0,256,(1200,16)).astype(np.float32)
                if k==0:x[rng.random(x.shape)<.8]=0
                if k==1:x/=np.float32(100)
                if k==2:x=rng.normal(size=x.shape).astype(np.float32)
                with (data/(name+'_train.fvecs')).open('wb') as f:
                    for row in x:
                        f.write(np.array([16],dtype='<i4').tobytes());f.write(row.astype('<f4').tobytes())
                alp.append({'Dataset':name,'Data compression ratio':2+k*.1,
                            'Compress time (s)':1+k*.1,'Build time (s)':1})
                for chain in (2,-1):
                    for i,c in enumerate(m.CODECS[1:]):
                        differential.append(dict(Dataset=name,Algorithm=c,ChainMaxLength=chain,
                            PayloadRatioF32=1.2+.2*i+.1*k, **{'CompressTime(s)':2+i+k*.2,'BuildTime(s)':1},
                            Vectors=1200,Dim=16))
            m.write_csv(root/'alp.csv',alp);m.write_csv(root/'diff.csv',differential)
            command=[sys.executable,str(Path(m.__file__)), 'calibrate','--data-dir',str(data),
                '--output-dir',str(root/'result'),'--datasets']+names+[
                '--samples','256','512','1024','--train-seeds','1','2','--tune-seeds','3',
                '--test-seeds','4','5','--references','16','--dimensions','8',
                '--ridges','.1','1','--space-weights','.25','.75',
                '--alp-oracle',str(root/'alp.csv'),'--differential-oracle',str(root/'diff.csv'),
                '--oracle-alp-raw-bytes','4']
            subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out=root/'result'
            rec=m.read_csv(out/'recommendations.csv')
            self.assertEqual(len(rec),3*3*2*2*2)
            self.assertEqual({int(r['Seed']) for r in rec},{4,5})
            self.assertEqual({r['Phase'] for r in rec},{'heldout_seed'})
            self.assertEqual(len(m.read_csv(out/'leave_dataset_out.csv')),len(rec))
            self.assertEqual(len(m.read_csv(out/'parameter_search.csv')),4*4*4)
            modelpath=out/'model_chainunbounded_all.json'
            model=json.loads(modelpath.read_text())
            self.assertEqual(model['settings']['sample_pool_size'],1024)
            self.assertEqual(model['training_datasets'],names)
            # Online path receives only a model and a data file, no oracle CSVs.
            (root/'alp.csv').unlink();(root/'diff.csv').unlink()
            infer=[sys.executable,str(Path(m.__file__)),'recommend','--model',str(modelpath),
                '--data-file',str(data/(names[0]+'_train.fvecs')),'--samples','512','--seed','4']
            result=json.loads(subprocess.check_output(infer))
            row=next(r for r in rec if r['Dataset']==names[0] and r['SampleCount']=='512'
                     and r['Seed']=='4' and r['ChainMaxLength']=='-1' and r['Pool']=='all')
            self.assertEqual(result['SelectedCodec'],row['SelectedCodec'])
            x,refs,n,dim,_,_=m.read_probe(data/(names[0]+'_train.fvecs'),512,4,model['settings'])
            features=m.extract_features(x,refs,dim,model['settings'])
            for feature in m.FEATURES:self.assertAlmostEqual(result['Features'][feature],features[feature])
            features['Dataset']='unseen_arbitrary_name'
            self.assertEqual(m.select(model,m.predict(model,features))[0],result['SelectedCodec'])
            # Restore label fixtures and ensure resume reuses exact probes.
            m.write_csv(root/'alp.csv',alp);m.write_csv(root/'diff.csv',differential)
            original=(out/'features.csv').read_bytes()
            subprocess.run(command+['--resume'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(original,(out/'features.csv').read_bytes())
            failed=subprocess.run(command+['--train-seeds','3'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn(b'disjoint',failed.stderr)


if __name__=='__main__':
    unittest.main(verbosity=2)
