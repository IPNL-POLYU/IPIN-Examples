% ------------------------------------------------------------------------
% Example for indoor positioning by time-difference-of-arrival (TDOA) measurements
% Estimated by iterative least-squares
%
% by GH.Zhang 2022/05/26
% guo-hao.zhang@connect.polyu.hk
% ------------------------------------------------------------------------
clc;
clear;
close all;

%% Estimation Setup
% Speed of light
c = 299792458;

% Import TOA measurement data and environment parameters
load("Example_data.mat");

% Operation time
T = size(data,2);

% Initialize agent approximate state (initial agent clock offset = 0)
x_a0 = x_a_initial;

% Select master beacon
j = 10;

%% Positioning with TOA measurements by iterative least-squares

% Position estimation on each epoch
for t = 1:T

    % Initialize agent state offset (initial agent clock offset update = 0)
    delta_x_a = [1e9; 1e9; 1e9];

    % Initialize iteration steps
    k = 0;

    % Initialize agent approximate state (initial agent clock offset = 0)
    x_a0 = x_a_initial;

    % Obtain range observation from master beacon
    d_a_j = c*data{t}(j).TOA;

    % Beacon numbers during positioning
    I = size(data{t},2);

    % iterative estimation until convergence or iteration step exceeds 
    % the limitation
    while norm(delta_x_a(1:3,1)) > 1e-3 && k < 50

        % clear variables from last estimation epoch
        clear d_a z_a z_a0 delta_z_a0 H_a0;

        % Establish variables for each beacon except master beacon
        for i = setdiff(1:I,j)

            % Obtain range observation by ranging measurement
            d_a(i,1) = c*data{t}(i).TOA;

            % Obtain differential range observation by range observations
            z_a(i,1) = d_a(i,1) - d_a_j;

            % Obtain approximate differential range information by beacon 
            % positions, master beacon position and agent approximate state
            z_a0(i,1) = norm(x_beacons(1:3,i) - x_a0) - ...
                        norm(x_beacons(1:3,j) - x_a0);

            % Calculate differential range offset from approximate 
            % differential range information to the received differential 
            % range observation 
            delta_z_a0(i,1) = z_a(i,1) - z_a0(i,1);

            % Calculate differential line-of-sight vector from each beacon 
            % to agent with respect to that from  master beacon to agent
            H_a0(i,1:3) = (x_a0 - x_beacons(1:3,i))'./norm(x_a0 - x_beacons(1:3,i)) -...
                          (x_a0 - x_beacons(1:3,j))'./norm(x_a0 - x_beacons(1:3,j));
        end
        
        % Exclude rows related to the master beacon in linear relatioships
        if size(delta_z_a0,1)>I-1
            delta_z_a0(j,:) = [];
            H_a0(j,:) = [];
        end

        % Establish linear relationships between differential range offset, 
        % measurement matrix, and state offset. Solve the state offset by 
        % least-sqaures estimation 
        delta_x_a = inv(H_a0'*H_a0)*H_a0'*delta_z_a0;

        % Update approximate agent state by the estimated state offset
        x_a0 = x_a0 + delta_x_a;

        % interation step update
        k = k + 1;
    end

    % Obtain agent position estimation from the converged state estimation
    x_a_est(1:3,t) = x_a0;
end

%% Visualize agent position estimation results
figure(1);
hold on;
plot(x_a_est(1,:),x_a_est(2,:),'g.','MarkerSize',15,'LineWidth',2);
plot(x_a(1),x_a(2),'kx','MarkerFaceColor','none','MarkerSize',20,'LineWidth',3);
plot(x_beacons(1,:),x_beacons(2,:),'k^','MarkerFaceColor','w','MarkerSize',10,'LineWidth',2);
xlabel('East (m)');
xlim([-60,60]);
ylabel('North (m)');
ylim([-60,60]);
axis equal
legend('Agent position estimation','Agent true position','Beacon positions')
title('TDOA Positioning');






















