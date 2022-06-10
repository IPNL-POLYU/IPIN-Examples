% ------------------------------------------------------------------------
% Simulation for TOA/AOA indoor positioning exmaple code
% Including the simulation of environment setup (agent, beacons position),
% TOA measurements as time difference, AOA measurements of
% elevation angle and azimuth angle
%
% by GH.Zhang 2022/06/10
% guo-hao.zhang@connect.polyu.hk
% ------------------------------------------------------------------------
clc;
clear;
close all;

%% Simulation Setup

% Speed of light (meter/second)
c = 299792458; 

% Agent true position in NED coordinate (meter)
x_a = [0,0,0]';

% Agent initial position in NED coordinate
% with Gaussian distribution N(0,2^2) in (meter)
x_a0 = [rand*2,rand*2,rand*2]';

% Simulation Duration (Second)
T = 300;

% Example beacon positions for simulation
% rows    - North, East, Down position (meter)
% columns - Values for different beacons
x_beacons(1:3,:) = [40,15,-15,-40,-50,-40,-15,15,40,50;...
                    30,47,47,30,0,-30,-47,-47,-30,0;...
                    0,5,10,5,0,5,10,5,0,5];

% Mean value of synchronized system time off-set between beacons and 
% agent (meter)
miu_delta_T_Ra = 10;

% Standard deviation of synchronized system time off-set between beacons 
% and agent (meter)
sigma_delta_T_Ra = 2;

% Standard deviation of TOA measurement error besides system time 
% off-set (meter)
sigma_TOA = 3;

% Standard deviation of AOA elevation angle measurement error (degree)
sigma_theta = 1;

% Standard deviation of AOA azimuth angle measurement error (degree)
sigma_psi = 1;

% Total number of beacons
I = size(x_beacons,2);

%% TOA Measurement Simulation

% Simulate measurements on individual operation epoch
for t = 1:T

    % Simulate system time off-set sychronized among beacons 
    % with Gaussian distribution (meter)
    cdt = randn*sigma_delta_T_Ra+miu_delta_T_Ra;

    % Individual beacon simulation
    for i = 1:I

        % Actual beacon-agent distance (meter)
        dist = norm(x_a-x_beacons(1:3,i));

        % Simulate TOA measurement noise with zero mean Gaussian
        % distribution (meter)
        dz = randn*sigma_TOA;

        % Simulate TOA measurement in time difference by beacon-agent 
        % distance, system time off-set and measurement noise (second)
        data{t}(i).TOA = (dist+dz+cdt)/c;

    end
end

%% AOA Measurement Simulation

% Simulate measurements on individual operation epoch
for t = 1:T

    % Individual beacon simulation
    for i = 1:I
        
        % Actual elevation angle of beacon observed by agent (degree)
        theta = asind((x_a(3,1)-x_beacons(3,i))/norm(x_a-x_beacons(1:3,i)));

        % Actual azimuth angle of beacon observed by agent (degree)
        psi = atan2d((x_beacons(2,i)-x_a(2,1)),(x_beacons(1,i)-x_a(1,1)));

        % Simulate AOA elevation angle measurement noise with zero mean
        % Guassian distribution (degree)
        d_theta = randn*sigma_theta;

        % Simulate AOA azimuth angle measurement noise with zero mean
        % Guassian distribution (degree)
        d_psi = randn*sigma_psi;

        % Simulate AOA elevation angle measurement with noise (degree)
        data{t}(i).AOA_theta = theta+d_theta;

        % Simulate AOA azimuth angle measurement with noise (degree)
        data{t}(i).AOA_psi = psi+d_psi;

    end
end



















