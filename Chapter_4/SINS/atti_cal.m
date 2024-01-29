
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
  
  %%%%%%%%%%��Ԫ���淶��%%%%%%%%%
  tmp_Q=sqrt(Q(1,1)^2+Q(2,1)^2+Q(3,1)^2+Q(4,1)^2);
  for kc=1:4
    Q(kc,1)=Q(kc,1)/tmp_Q;    
  end
 
%%%%%%%%%%��ȡ��̬����%%%%%%%%%
  Cbn=[Q(2,1)^2+Q(1,1)^2-Q(4,1)^2-Q(3,1)^2, 2*(Q(2,1)*Q(3,1)+Q(1,1)*Q(4,1)), 2*(Q(2,1)*Q(4,1)-Q(1,1)*Q(3,1));
       2*(Q(2,1)*Q(3,1)-Q(1,1)*Q(4,1)), Q(3,1)^2-Q(4,1)^2+Q(1,1)^2-Q(2,1)^2,  2*(Q(3,1)*Q(4,1)+Q(1,1)*Q(2,1));
       2*(Q(2,1)*Q(4,1)+Q(1,1)*Q(3,1)), 2*(Q(3,1)*Q(4,1)-Q(1,1)*Q(2,1)), Q(4,1)^2-Q(3,1)^2-Q(2,1)^2+Q(1,1)^2]; 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %����̬(��������������� (����Ҫ�Է�ĸΪ���������д�������2005��4��3�գ�Ŀǰ��û���޸�)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  attiN(1,1)=atan(-Cbn(1,3)/Cbn(3,3));
  attiN(2,1)=atan(Cbn(2,3)/sqrt(Cbn(2,1)*Cbn(2,1)+Cbn(2,2)*Cbn(2,2)));
  attiN(3,1)=atan(Cbn(2,1)/Cbn(2,2));
    %��λ������

  %�����ж�
  attiN(1,1)=attiN(1,1)*180.0/pi;
  attiN(2,1)=attiN(2,1)*180.0/pi;
  attiN(3,1)=attiN(3,1)*180.0/pi;
    % ��λ����

  if(Cbn(2,2)<0 ) 
   attiN(3,1)=180.0+attiN(3,1);
  else 
   if(Cbn(2,1)<0) attiN(3,1)=360.0+attiN(3,1); end
  end
    %����Ƕȣ���λ���ȣ�

  if(Cbn(3,3)<0)  
   if(Cbn(1,3)>0) attiN(1,1)=-(180.0-attiN(1,1)); end
   if(Cbn(1,3)<0) attiN(1,1)=(180.0+attiN(1,1)); end
  end
    %����Ƕȣ���λ���ȣ�


