
function [attiN]=atti_cal(T,Wibb,attiN)
 
  roll=attiN(1,1)*pi/180.0;pitch=attiN(2,1)*pi/180.0;head=attiN(3,1)*pi/180.0;
  
  Q=[cos(head/2)*cos(pitch/2)*cos(roll/2)+sin(head/2)*sin(pitch/2)*sin(roll/2);
     cos(head/2)*sin(pitch/2)*cos(roll/2)+sin(head/2)*cos(pitch/2)*sin(roll/2);
     cos(head/2)*cos(pitch/2)*sin(roll/2)-sin(head/2)*sin(pitch/2)*cos(roll/2);
     -1.0*sin(head/2)*cos(pitch/2)*cos(roll/2)+cos(head/2)*sin(pitch/2)*sin(roll/2)];

  WnbbA=Wibb*T;
  
  WnbbX=[0,          -WnbbA(1,1), -WnbbA(2,1), -WnbbA(3,1);
         WnbbA(1,1),  0,           WnbbA(3,1), -WnbbA(2,1);
         WnbbA(2,1), -WnbbA(3,1),   0,          WnbbA(1,1);
         WnbbA(3,1),  WnbbA(2,1),  -WnbbA(1,1),   0         ];

  Q=Q+0.5*WnbbX*Q;
  
  %%%%%%%%%%四元数规范化%%%%%%%%%
  tmp_Q=sqrt(Q(1,1)^2+Q(2,1)^2+Q(3,1)^2+Q(4,1)^2);
  for kc=1:4
    Q(kc,1)=Q(kc,1)/tmp_Q;    
  end
 
%%%%%%%%%%获取姿态矩阵%%%%%%%%%
  Cbn=[Q(2,1)^2+Q(1,1)^2-Q(4,1)^2-Q(3,1)^2, 2*(Q(2,1)*Q(3,1)+Q(1,1)*Q(4,1)), 2*(Q(2,1)*Q(4,1)-Q(1,1)*Q(3,1));
       2*(Q(2,1)*Q(3,1)-Q(1,1)*Q(4,1)), Q(3,1)^2-Q(4,1)^2+Q(1,1)^2-Q(2,1)^2,  2*(Q(3,1)*Q(4,1)+Q(1,1)*Q(2,1));
       2*(Q(2,1)*Q(4,1)+Q(1,1)*Q(3,1)), 2*(Q(3,1)*Q(4,1)-Q(1,1)*Q(2,1)), Q(4,1)^2-Q(3,1)^2-Q(2,1)^2+Q(1,1)^2]; 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %求姿态(横滚、俯仰、航向） (还需要对分母为零的情况进行处理！！！2005年4月3日，目前还没有修改)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  attiN(1,1)=atan(-Cbn(1,3)/Cbn(3,3));
  attiN(2,1)=atan(Cbn(2,3)/sqrt(Cbn(2,1)*Cbn(2,1)+Cbn(2,2)*Cbn(2,2)));
  attiN(3,1)=atan(Cbn(2,1)/Cbn(2,2));
    %单位：弧度

  %象限判断
  attiN(1,1)=attiN(1,1)*180.0/pi;
  attiN(2,1)=attiN(2,1)*180.0/pi;
  attiN(3,1)=attiN(3,1)*180.0/pi;
    % 单位：度

  if(Cbn(2,2)<0 ) 
   attiN(3,1)=180.0+attiN(3,1);
  else 
   if(Cbn(2,1)<0) attiN(3,1)=360.0+attiN(3,1); end
  end
    %航向角度（单位：度）

  if(Cbn(3,3)<0)  
   if(Cbn(1,3)>0) attiN(1,1)=-(180.0-attiN(1,1)); end
   if(Cbn(1,3)<0) attiN(1,1)=(180.0+attiN(1,1)); end
  end
    %横滚角度（单位：度）


