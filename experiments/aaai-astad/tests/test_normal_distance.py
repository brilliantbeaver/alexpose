import numpy as np
import pandas as pd
import aaai_gait.normal_distance as nd

def test_normal_distance_covariance_uses_only_normal_training_sources(monkeypatch):
    records=[]
    for source,label,split in [('a','normal','train'),('b','normal','train'),('c','stroke','train'),('d','normal','val'),('e','normal','test'),('f','stroke','test')]:
        records.append(dict(source_id=source,sequence_id=source,label=label,split=split))
    meta=pd.DataFrame(records);z=np.arange(12,dtype=float).reshape(6,2)
    monkeypatch.setattr(nd,'embed_windows',lambda *args,**kwargs:z)
    seen=[]
    class Covariance:
        def fit(self,values):seen.append(values.copy());return self
        def mahalanobis(self,values):return values[:,0]+1
    monkeypatch.setattr(nd,'LedoitWolf',Covariance)
    source,result=nd.normal_distance(None,None,meta)
    np.testing.assert_array_equal(seen[0],z[:2])
    assert result['n_normal_training_sources']==2
    assert result['finite_threshold'] is False
    assert len(source)==6
