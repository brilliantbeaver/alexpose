import numpy as np
from worldmodels.extended_data import gate_person

def test_person_gate_scales_bbox_and_preserves_unresolved():
    xy=np.ones((33,2))*50;conf=np.ones(33)
    bbox={'left':0,'top':0,'width':100,'height':100};info={'width':200,'height':200}
    assert gate_person(xy,conf,bbox,info,100,100)[1]=='inside_box'
    xy[23:25]=80
    gated,reason=gate_person(xy,conf,bbox,info,100,100)
    assert reason=='outside_annotated_person_box' and not gated.any()
    gated,reason=gate_person(xy,conf,bbox,info,100,50)
    assert reason=='unresolved_aspect_ratio' and gated.all()
