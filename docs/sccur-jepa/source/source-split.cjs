// Schematic assignments only. A and B identify source recordings; digits identify clips.
module.exports=function sourceSplit(p,s,x,y,w,h){
  const sx=w/21.64,sy=h/1.95,u=Math.min(sx,sy),navy='1F3864',teal='00796D',slate='435563';
  function label(t,a,b,ww,hh,pt,more={}){
    s.addText(t,{x:x+a*sx,y:y+b*sy,w:ww*sx,h:hh*sy,margin:0,fontFace:'Calibri',fontSize:pt*u,color:navy,valign:'mid',...more});
  }
  function chip(a,b,id){
    const A=id[0]==='A',color=A?navy:teal;
    s.addShape(A?p.ShapeType.rect:p.ShapeType.roundRect,{x:x+a*sx,y:y+b*sy,w:1.00*sx,h:.48*sy,
      rectRadius:.06*u,radius:.06*u,fill:{color:A?'EAF0F9':'E8F4EE'},line:{color,width:1.45*u}});
    label(id,a,b,1,.48,23,{bold:true,color,align:'center'});
  }
  function side(a,title,train,test,ending){
    label(title,a,0,10.48,.48,27,{bold:true});
    label('Train',a,.65,1.10,.48,24,{bold:true,color:slate});
    train.forEach((id,i)=>chip(a+1.27+i*1.16,.65,id));
    label('Test',a+4.12,.65,1.02,.48,24,{bold:true,color:slate});
    test.forEach((id,i)=>chip(a+5.24+i*1.16,.65,id));
    label(ending,a,1.32,10.45,.57,24,{color:slate});
  }
  side(.02,'Study A · Split clips',['A1','B1'],['A2','B2'],'Test clips can come from familiar recordings.');
  s.addShape(p.ShapeType.line,{x:x+10.75*sx,y:y+.10*sy,w:0,h:1.67*sy,line:{color:'C9D6DF',width:1.35*u}});
  side(11.00,'Study B · Hold out whole recordings',['A1','A2'],['B1','B2'],'Test clips come from a new recording.');
  label('Illustrative clip IDs',17.98,.02,3.62,.40,20,{italic:true,color:slate,align:'right'});
};
