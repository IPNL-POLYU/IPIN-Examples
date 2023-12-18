clear;
close all;

% global parameter
iteration_num = 10000;
wifi_noise = 15;

% trajectory generation
radius = 500; % radius
center = [0, 0]; % origin

theta_tmp = linspace(0, 2*pi, iteration_num); % angle slice
wifi_x = center(1) + radius * cos(theta_tmp)+ wifi_noise*randn(1,iteration_num); % wifi figerprinting position x
wifi_y = center(2) + radius * sin(theta_tmp)+ wifi_noise*randn(1,iteration_num); % wifi figerprinting position y
gt_x = center(1) + radius * cos(theta_tmp); % gt x
gt_y = center(2) + radius * sin(theta_tmp); % gt y
          
% calculate PDR parameter
L=zeros(1,iteration_num-1);
theta=zeros(1,iteration_num-1);

for i = 2:iteration_num
    L(i-1) = sqrt((gt_x(i) - gt_x(i-1))*(gt_x(i) - gt_x(i-1)) + (gt_y(i) - gt_y(i-1))*(gt_y(i) - gt_y(i-1)));
    theta(i-1) = atan((gt_y(i) - gt_y(i-1))/(gt_x(i) - gt_x(i-1)));
end

L = L + 0.5*randn(1,iteration_num-1); 
theta = theta + 0.2*randn(1,iteration_num-1);  

figure(1)
plot(L)
ylim([-10 10]);
figure(2)
plot(theta)

% calculate PDR trajectory
PDR_pos = zeros(2,iteration_num-1);
PDR_pos(:,1) = [radius 0];
for i = 2:iteration_num/2
    PDR_pos(1,i) = PDR_pos(1,i-1) - L(i)*cos(theta(i));
    PDR_pos(2,i) = PDR_pos(2,i-1) - L(i)*sin(theta(i));
end
for i = iteration_num/2+1:iteration_num-1
    PDR_pos(1,i) = PDR_pos(1,i-1) + L(i)*cos(theta(i));
    PDR_pos(2,i) = PDR_pos(2,i-1) + L(i)*sin(theta(i));
end

% EKF-based WiFi-PDR Integration

% Step 1: Initialization
% Define state vector: [x, y, vx, vy, theta]
X = zeros(4, iteration_num-1);  % Initial state estimate
P = eye(4);       % Initial error covariance matrix
X(:,1) = [0 0 0 0];

% Step 2: Extended Kalman Filter Algorithm
% Prediction and Correction steps
for k = 2:iteration_num-1
    % Step 2: System Model
    A = [1 0 cos(X(4, k-1)) -X(3, k-1)*sin(X(4, k-1));  % State transition matrix
        0 1 sin(X(4, k-1)) X(3, k-1)*cos(X(4, k-1));
        0 0 1 0;
        0 0 0 1];
    Q = diag([0.01 0.01 0.01 0.01]);  % Process noise covariance matrix

    % Step 3: Wi-Fi Measurement Model
    % Assume a linear measurement model relating WiFi signal strengths to user's position
    H = [1 0 0 0;  % Measurement matrix
         0 1 0 0;
         0 0 1 0;
         0 0 0 1];
    R = diag([0.5 0.5 0.001 0.001]);  % Measurement noise covariance matrix
    % PDR Update (Prediction step)
    % Update user's position based on PDR information
    X(:, k) = A * X(:, k-1);  % x state update
    
    % Update the error covariance matrix based on the system model
    P = A * P * A' + Q;
    
    % WiFi Fingerprint Measurement Update (Correction step)
    % Assume WiFi fingerprinting positioning is available at each step
    z = [wifi_x(k-1);  % WiFi signal strength measurement in x-direction
         wifi_y(k-1); % WiFi signal strength measurement in y-direction
         L(k-1);
         theta(k-1)];

    % Compute measurement residuals
    y = z - H * X(:, k);
     
    % Compute Kalman gain
    S = H * P * H' + R;       % Innovation covariance
    K = P * H' / (S);         % Kalman gain
    
    % Update state estimate and error covariance matrix
    X(:, k) = X(:, k) + K * y;      % Updated state estimate
    P = (eye(4) - K * H) * P;       % Updated error covariance matrix    
end

figure (3)
for i = 1:iteration_num
    error_wifi(i) = sqrt((wifi_x(i) - gt_x(i))*(wifi_x(i) - gt_x(i)) + (wifi_y(i) - gt_y(i))*(wifi_y(i) - gt_y(i)));
end
for i = 1:iteration_num-1
    error_pdr(i) = sqrt((PDR_pos(1,i) - gt_x(i))*(PDR_pos(1,i) - gt_x(i)) + (PDR_pos(2,i) - gt_y(i))*(PDR_pos(2,i) - gt_y(i)));
    error_EKF(i) = sqrt((X(1,i) - gt_x(i))*(X(1,i) - gt_x(i)) + (X(2,i) - gt_y(i))*(X(2,i) - gt_y(i)));
end

fprintf('error_wifi=%f\n', mean(error_wifi))
fprintf('error_wifi_std=%f\n', std(error_wifi))
fprintf('\n')

fprintf('error_pdr=%f\n', mean(error_pdr))
fprintf('error_pdr_std=%f\n', std(error_pdr))
fprintf('\n')

fprintf('error_EKF=%f\n', mean(error_EKF))
fprintf('error_EKF_std=%f\n', std(error_EKF(iteration_num/10:iteration_num-1)))
fprintf('\n')

plot(error_wifi,'b-','LineWidth',3 )
hold on;
plot(error_pdr,'r-','LineWidth',3 )
hold on;
plot(error_EKF,'g-','LineWidth',3 )
hold on;

grid on;
hold on;
ax = gca;
ax.FontSize = 24; 
xlabel('time (epoch)');
ylabel('error (meters)');
legend('\fontsize{24} WIFI fingerprint','\fontsize{24} PDR','\fontsize{24} EKF');
title('\fontsize{24} Error')

% show trajectory
figure (4)
scatter(wifi_x, wifi_y, 'filled', 'MarkerFaceColor',[0 0 1]);
hold on;
scatter(PDR_pos(1,:), PDR_pos(2,:), 'filled', 'MarkerFaceColor',[1 0 0]);
hold on;
scatter(X(1,:), X(2,:), 'filled', 'MarkerFaceColor',[0 1 0]);
hold on;
scatter(gt_x, gt_y, 'filled', 'MarkerFaceColor',[0 0 0],'LineWidth',0.1);
hold on;
axis equal;
grid on;
ax = gca;
ax.FontSize = 24; 
xlabel('x(m)');
ylabel('y(m)');
legend('\fontsize{24} WIFI fingerprint','\fontsize{24} PDR','\fontsize{24} EKF', '\fontsize{24} GT');
title('\fontsize{24} Trajectory')

% figure(5)
% scatter(PDR_pos(1,:), PDR_pos(2,:), 'filled');
% axis equal;