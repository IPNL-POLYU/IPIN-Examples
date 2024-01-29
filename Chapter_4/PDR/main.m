clear;clc
load imu_data.mat;
load GT.mat;
trial = imu_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yaw = trial(:,5);
NN = length(yaw);
head = zeros(NN,1);

for i=1:NN
    
    if yaw(i,1)>=0&&yaw(i,1)<=90
        head(i,1) = 90 - yaw(i,1);
    end
    
    if yaw(i,1)>=90&&yaw(i,1)<=180
        head(i,1) = 90 - yaw(i,1);
    end
    
    if yaw(i,1)>=180&&yaw(i,1)<=270
       head(i,1) = 90 - yaw(i,1); 
    end
    
    if yaw(i,1)>=270&&yaw(i,1)<=360
        head(i,1) = 450 - yaw(i,1);
    end
    
end

zoneA = find(head>=0    & head<90);
zoneB = find(head>=90   & head<180);
zoneC = find(head>=-180 & head<-90);
zoneD = find(head>=-90  & head<0);

head(zoneA,2:3) = [head(zoneA,1) head(zoneA,1)];
head(zoneB,2:3) = [180-head(zoneB,1) head(zoneB,1)];
head(zoneC,2:3) = [-head(zoneC,1)-180 -head(zoneC,1)];
head(zoneD,2:3) = [head(zoneD,1) -head(zoneD,1)];

acc = sqrt(trial(:,2).^2 + trial(:,3).^2 + trial(:,4).^2);
acc = acc - mean(acc);

%% Bandpass filter
fs = 100;
f1 = 0.75;
f2 = 2.75;
Wn = [f1 f2]/(fs/2);
N = 4;

[a,b] = butter(N,Wn);
bandPass = filtfilt(a,b,acc);

%% Find peaks
[PkValue, PeakLocation] = findpeaks(bandPass, 'MINPEAKHEIGHT', 0.25);

%% Time interval
PkValue(:,2) = trial(PeakLocation,1);
PkValue(2:end,2) = PkValue(2:end,2)-PkValue(1:end-1,2);
index = find(PkValue(:,2)<400);

if isempty(index) == 0
    pos_del = [];
    for k=1:length(index)
        temp = index(k);
        if PkValue(temp,1) <= PkValue(temp-1,1)
            pos_del = [pos_del; temp];
        else
            pos_del = [pos_del; temp-1];
        end
    end
    PeakLocation(pos_del) = [];
    PkValue(pos_del,:) = [];
end
StepCount = length(PeakLocation);
pdr = zeros(StepCount, 2);

height = 1.70;
for t = 2:StepCount
    
        pos_start = PeakLocation(t-1);
        pos_end = PeakLocation(t);
        
        YawSin = mean(head(pos_start:pos_end,2));
        YawCos = mean(head(pos_start:pos_end,3));
        
        StepFreq = 1000/PkValue(t,2);
        
        StepLength = 0.7 + 0.371*(height-1.75) + 0.227*(StepFreq-1.79)*height/1.75;
       
        pdr(t,1) = pdr(t-1,1) + StepLength * cosd(YawCos);
        pdr(t,2) = pdr(t-1,2) + StepLength * sind(YawSin);
        
end

N1 = 107;

figure();
plot(GT(1:N1,1),GT(1:N1,2),'k','linewidth',2);
hold on;
plot(pdr(1:N1,1),pdr(1:N1,2),'k--','linewidth',2);
hold on;
xlabel('X[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('Y[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1); 
set(gca,'Fontsize',11); 
hl=legend('Ground Truth','PDR','Location','southeast','FontSize',12,'FontWeight','bold','FontName','Segoe UI Semilight');
% hl=legend('S-PDR','A-SLAM','Proposed-ISE','Proposed-BE','FontSize',13);
set(hl,'box','off');

figure();
subplot(2,1,1);
aa = pdr(1:N1,1)-GT(1:N1,1);
plot((1:N1)*0.56,aa,'k','linewidth',2);
xlabel('Time[s]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('X[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
title('Positioning Error','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1);
set(gca,'Fontsize',11); 

subplot(2,1,2);
aa = pdr(1:N1,2)-GT(1:N1,2);
plot((1:N1)*0.56,aa,'k','linewidth',2);
xlabel('Time[s]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
ylabel('Y[m]','Fontsize',10,'FontWeight','bold','FontName','Segoe UI Semilight');
set(gca,'Linewidth',1.1); 
set(gca,'Fontsize',11); 