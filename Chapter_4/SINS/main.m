clear;clc
load imu_data.mat;
load GT.mat;

NN = size(imu_data,1);

latitude = 22.30409 * pi / 180;
g=9.7803*(1+0.0053024*((sin(latitude))^(2))-0.000005*((sin(2*latitude))^(2)));

pose = zeros(NN,9);

atti = [-1;34;278];
velo = zeros(3,1);
posi = GT(1,1:3)';

pose(1,:) = [atti',velo',posi'];

for i=1:NN-1
    
    T = imu_data(i+1,1) - imu_data(i,1);
    
    Wibb = imu_data(i,5:7)';
    Accb = [imu_data(i,2),imu_data(i,3),imu_data(i,4)]';
    
    [posi]=posi_cal(T,velo,posi);
    
    [velo]=velo_cal(T,Accb,atti,velo,g);
    
    [atti]=atti_cal(T,Wibb,atti);
    
    pose(i+1,:) = [atti',velo',posi'];
end


figure();
plot(GT(:,1),GT(:,2),'k','linewidth',2);
hold on;
plot(pose(:,7),pose(:,8),'k--','linewidth',2);
hold on;
xlabel('X[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('Y[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1); 
set(gca,'Fontsize',11);
hl=legend('Ground Truth','Inertial Navigation','Location','southwest','FontSize',12,'FontWeight','bold','FontName','Segoe UI Semilight');
% hl=legend('S-PDR','A-SLAM','Proposed-ISE','Proposed-BE','FontSize',13);
set(hl,'box','off');

figure();
subplot(2,1,1);
aa = sqrt((pose(:,7)-GT(:,1)).^2+(pose(:,8)-GT(:,2)).^2);
plot((1:NN)*0.01,aa,'k','linewidth',2);
xlabel('Time[s]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('Horizontal[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
title('Positioning Error','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1); 
set(gca,'Fontsize',11);

subplot(2,1,2);
aa = abs(pose(:,9)-GT(:,3));
plot((1:NN)*0.01,aa,'k','linewidth',2);
xlabel('Time[s]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('Vertical[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1); 
set(gca,'Fontsize',11);