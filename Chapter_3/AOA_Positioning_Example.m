% ------------------------------------------------------------------------
% Example for indoor positioning by time-of-arrival (TOA) measurements
% Estimated by iterative least-squares and weighted least-squares
%
% by GH.Zhang 2022/05/26
% guo-hao.zhang@connect.polyu.hk
% ------------------------------------------------------------------------
clc;
clear;
close all;

%% Estimation Setup

% Import TOA measurement data and environment parameters
load("Example_data.mat");

% Operation time
T = size(data,2);

%% Positioning with TOA measurements by iterative least-squares

% Position estimation on each epoch
for t = 1:T
  
    % Initialize agent state offset (initial agent clock offset update = 0)
    delta_x_a = [1e9; 1e9; 1e9; 0];

    % Initialize iteration steps
    k = 0;

    % Initialize agent approximate state (initial agent clock offset = 0)
    x_a0 = x_a_initial;

    % Beacon numbers during positioning
    I = size(data{t},2);

    % Iterative estimation until convergence or iteration step exceeds 
    % the limitation
    while norm(delta_x_a(1:3,1)) > 1e-3 && k < 50

        % Clear variables from last estimation epoch
        clear z_a z_a0 delta_z_a0 H_a0;

        % Establish variables for each beacon
        for i = 1:I

            % Obtain trigonometric observation by AOA measurement
            z_a(2*i-1,1) = sind(data{t}(i).AOA_theta);
            z_a(2*i,1) = tand(data{t}(i).AOA_psi);

            % Obtain approximate trigonometric information by beacon 
            % positions and agent approximate position
            z_a0(2*i-1,1) = (x_a0(3,1)-x_beacons(3,i))/norm(x_a0-x_beacons(1:3,i));
            z_a0(2*i,1) = (x_beacons(2,i)-x_a0(2,1))/(x_beacons(1,i)-x_a0(1,1));

            % Calculate the offset from approximate trigonometric 
            % information to the received trigonometric observation
            delta_z_a0(2*i-1:2*i,1) = z_a(2*i-1:2*i,1) - z_a0(2*i-1:2*i,1);
            
            % Partial derivatives of the observation model corresponding to
            % elevation angle
            f_i_n = -(x_a0(3,1)-x_beacons(3,i))*(x_a0(1,1)-x_beacons(1,i))/norm(x_a0-x_beacons(1:3,i))^3;
            f_i_e = -(x_a0(3,1)-x_beacons(3,i))*(x_a0(2,1)-x_beacons(2,i))/norm(x_a0-x_beacons(1:3,i))^3;
            f_i_d = ((x_a0(1,1)-x_beacons(1,i))^2+(x_a0(2,1)-x_beacons(2,i))^2)/norm(x_a0-x_beacons(1:3,i))^3;
            
            % Partial derivatives of the observation model corresponding to
            % azimuth angle
            g_i_n = -(x_a0(2,1)-x_beacons(2,i))/(x_a0(1,1)-x_beacons(1,i))^2;
            g_i_e = (x_a0(1,1)-x_beacons(1,i))/(x_a0(1,1)-x_beacons(1,i))^2;
            g_i_d = 0;

            % Establish the block matrix of the measurement matrix
            % corresponding to each beacon
            H_a0(2*i-1,1:3) = [f_i_n,f_i_e,f_i_d];
            H_a0(2*i,1:3) = [g_i_n,g_i_e,g_i_d];
        end

        % Establish linear relationships between trigonometric observation 
        % offset, measurement matrix, and state offset. Solve the state 
        % offset by least-sqaures estimation 
        delta_x_a = inv(H_a0'*H_a0)*H_a0'*delta_z_a0;

        % Update approximate agent state by the estimated state offset
        x_a0 = x_a0 + delta_x_a;

        % interation step update
        k = k + 1;
    end

    % Obtain agent position estimation from the converged state estimation
    x_a_est(1:3,t) = x_a0(1:3,1);
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
title('AOA Positioning');























