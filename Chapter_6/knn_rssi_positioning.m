
%% step0: configuration
close; clc; clear; 
% set a rectangular shaped study area
area_xlim = [0, 50]; % in meters
area_ylim = [0, 25]; % in meters

% generate evenly spaced grid locations throughout the study area
grid_size = 1; % in meters
xgrid_positions = area_xlim(1) : grid_size : area_xlim(2);
ygrid_positions = area_ylim(1) : grid_size : area_ylim(2);

num_xgrids = length(xgrid_positions);
num_ygrids = length(ygrid_positions);

% A minimum number of 3 anchors are needed for 2D positioning.
% [1] Liu, H., Darabi, H., Banerjee, P., & Liu, J. (2007). 
% Survey of wireless indoor positioning techniques and systems. 
% IEEE Transactions on Systems, Man and Cybernetics 
% Part C: Applications and Reviews, 37(6), 1067–1080. 
% https://doi.org/10.1109/TSMCC.2007.905750
num_anchors = 3;
% set anchor's 2D positions (x, y)
anchor_positions = [
    5, 5;   % position of anchor 1
    45, 5;   % position of anchor 2
    25, 20];  % position of anchor 3

% use a 3-dim matrix ([x, y, anchor]) to store fingerprint maps
fingerprint_map = zeros(num_xgrids, num_ygrids, num_anchors);

% standard deviation of white Gaussian noise in RSSI measurements
noise_std = 0.5; % in dB

% parameters for the log-distance path-loss model, 
% see: https://en.wikipedia.org/wiki/Log-distance_path_loss_model
PL0 = -35; % the path loss in (dB) at the reference distance (1 meter)
gamma = 2; % the path loss exponent

%% step1: generate fingerprint maps
for i = 1:num_xgrids
    for j = 1 : num_ygrids
        % consider the current grid location
        grid_position = [xgrid_positions(i), ygrid_positions(j)];
        % repeat the position vector "num_anchors" of times,
        % since we need ranging to all these anchors
        grid_positions = repmat(grid_position, num_anchors, 1);
        % compute the Euclidean distances from this grid to each anchor
        range_to_anchors = sqrt(sum((anchor_positions - grid_positions).^2, 2));
        
        % generate RSSI measurements based on the log-distance pathloss
        % model, see: https://en.wikipedia.org/wiki/Log-distance_path_loss_model
        % PL(d) = PL0 - 10*gamma*log10(d)
        % [2] Bahl, P., & Padmanabhan, V. N. (2000). 
        % RADAR: An in-building RF-based user location and tracking system. 
        % Proceedings - IEEE INFOCOM, 2(c), 775–784. 
        % https://doi.org/10.1109/infcom.2000.832252
        rssi_of_anchors = PL0 - 10 * gamma * log10(range_to_anchors);
        % save the generated fingerprint (RSSI vector) to our map
        fingerprint_map(i, j, :) = rssi_of_anchors;
    end
end

%% step2: visualize fingerprint maps for each anchor
figure;
for anchor_id = 1:num_anchors
    % get the fingerprint map for each anchor
    map_anchor = reshape(fingerprint_map(:,:,anchor_id), num_xgrids, num_ygrids);
    
    % create a new subplot to do drawing
    subplot(num_anchors,1,anchor_id);
    % show the fingerprint for a given anchor as a heatmap
    imagesc(xgrid_positions, ygrid_positions, map_anchor')
    set(gca,'YDir','normal')
    colorbar()
    hold on
    
    % plot the anchor position
    plot(anchor_positions(anchor_id,1), anchor_positions(anchor_id,2), 'r^', 'MarkerFaceColor', 'r');
    hold off
   
    title(strcat('anchor', num2str(anchor_id)))
    xlabel('x [m]')
    ylabel('y [m]')
    axis('equal')
    axis('tight')
end

%% step3: static point positioning based on the K-nearest neighbor (KNN)
% using new RSSI measurements at unknown locations
% [3] Sun, Y., Liu, M., & Meng, M. Q.-H. H. Q. H. (2014). 
% WiFi signal strength-based robot indoor localization. 
% 2014 IEEE International Conference on Information and Automation, ICIA 2014, July, 250–256. 
% https://doi.org/10.1109/ICInfA.2014.6932662

% 3-1: generate noisy RSSI measurements for each anchor
test_position = [20, 15]; % Your can change the position to any location within the study area
% repeat the position vector "num_anchors" of times 
test_position_repeat = repmat(test_position, num_anchors, 1);
% compute the Euclidean distances from this grid to each anchor
range_to_anchors = sqrt(sum((anchor_positions - test_position_repeat).^2, 2));
% generate RSSI measurements based on the log-distance pathloss model
rssi_of_anchors = PL0 - 10 * gamma * log10(range_to_anchors);
% add additive white Gaussian noise to the truth RSSI to
% simulate noisy measurements on hardware
rssi_of_anchors_noisy = rssi_of_anchors + noise_std * randn(num_anchors,1);

% 3-2: do KNN positioning
% compute fingerprint mismatches (matching errors) for each grid candidate
matching_error_map = zeros(num_xgrids, num_ygrids);
for i = 1:num_xgrids
    for j = 1 : num_ygrids
        rssi_fingerprint = reshape(fingerprint_map(i, j, :), [], 1);
        rssi_error_per_anchor = sqrt(sum((rssi_of_anchors_noisy - rssi_fingerprint).^2, 2));
        matching_error_map(i,j) = mean(rssi_error_per_anchor);
    end
end

% find the grid candidate that has minimum matching errors (i.e., nearest neighbour)
[min_error, knn_index] = min(matching_error_map(:));
% remap the found 1D index to a 2D grid index
xgrid_index = mod(knn_index-1, num_xgrids)+1;
ygrid_index = floor((knn_index-1)/num_xgrids)+1;

% take the nearest grid location as our KNN position estimate 
knn_position = [xgrid_positions(xgrid_index), ygrid_positions(ygrid_index)];
 
% 3-3: visualize the fingerprint matching errors and position results.
figure;
% visualize the fingerprint matching errors as a heatmap. The brightest color 
% represents the median of all matching errors. This allows for clearer visualization.
imagesc(xgrid_positions, ygrid_positions, matching_error_map', [0 median(matching_error_map(:))])
set(gca,'YDir','normal')
% colormap(gray)
colorbar()
hold on

% plot the ground truth and KNN-estimated positions
plot(test_position(1), test_position(2), 'g.', 'MarkerSize', 20);
plot(knn_position(1), knn_position(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
% draw a line from the current ground truth position to the estimated
% position, showing the position error
line([test_position(1) knn_position(1)], [test_position(2) knn_position(2)], 'Color','red', 'LineWidth', 2)
hold off

title('Fingerprint matching errors')
legend('Ground truth', 'Position estimate', 'Position error', 'Location', 'northoutside', 'NumColumns', 3);
xlim(area_xlim)
ylim(area_ylim)
xlabel('x [m]')
ylabel('y [m]')
axis('equal')
axis('tight')

%% animation of continuous positioning
sim_time = 10;
sim_num_samples = 50;
T = linspace(0, sim_time, sim_num_samples);

% generate a simulated trajectory in the shape of a semicircle
sim_traj_x = 15 * (1 + cos(0.1 * pi * T)) + 10;
sim_traj_y = 15 * sin(0.1 * pi * T) + 1;
sim_positions = [sim_traj_x; sim_traj_y]';

% Initialize a variable to store estimated positions by KNN 
knn_positions = zeros(sim_num_samples, 2);
 
% create a figure object
figure;
ax = gca;
% do KNN positioning at each timestamp, and show the result
for id = 1:length(T)
    % consider the current test position
    test_position = sim_positions(id, :);
    test_position_repeat = repmat(test_position, num_anchors, 1);
    % compute the Euclidean distances from this grid to each anchor
    range_to_anchors = sqrt(sum((anchor_positions - test_position_repeat).^2, 2));

    % generate RSSI measurements based on the log-distance pathloss model
    rssi_of_anchors = PL0 - 10 * gamma * log10(range_to_anchors);
    % add additive white Gaussian noise to the truth RSSI to
    % simulate noisy measurements on hardware
    rssi_of_anchors_noisy = rssi_of_anchors + noise_std * randn(num_anchors,1);
    
    % compute fingerprint mismatches (matching errors) for each grid candidate
    matching_error_map = zeros(num_xgrids, num_ygrids);
    for i = 1:num_xgrids
        for j = 1 : num_ygrids
            rssi_fingerprint = reshape(fingerprint_map(i, j, :), [], 1);
            rssi_error_per_anchor = sqrt(sum((rssi_of_anchors_noisy - rssi_fingerprint).^2, 2));
            matching_error_map(i,j) = mean(rssi_error_per_anchor);
        end
    end
    
    % find the grid candidate that has minimum matching errors (i.e., nearest neighbour)
    [min_error, knn_index] = min(matching_error_map(:));
    % remap the found 1D index to a 2D grid index
    xgrid_index = mod(knn_index-1, num_xgrids)+1;
    ygrid_index = floor((knn_index-1)/num_xgrids)+1;
    
    % take the nearest grid location as our KNN position estimate 
    knn_positions(id, :) = [xgrid_positions(xgrid_index), ygrid_positions(ygrid_index)];

    % visualize the fingerprint matching errors as a heatmap. The brightest color 
    % represents the median of all matching errors. This allows for clearer visualization.
    imagesc(ax, xgrid_positions, ygrid_positions, matching_error_map', [0 median(matching_error_map(:))])
    set(ax,'YDir','normal')
%     colormap(gray)
    colorbar()
    hold on
    % plot the ground truth and KNN-estimated positions until the current timestep
    plot(ax, sim_positions(1:id,1), sim_positions(1:id,2), 'g.', 'MarkerSize', 20)
    plot(ax, knn_positions(1:id,1), knn_positions(1:id,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2)
    % draw a line from the current ground truth position to the estimated
    % position, showing the position error
    line(ax, [sim_positions(id,1) knn_positions(id, 1)], [sim_positions(id,2) knn_positions(id, 2)], 'Color','red', 'LineWidth', 2)
    hold off
    
    legend('Ground truth', 'Position estimate', 'Position error', 'Location', 'northoutside', 'NumColumns', 3);
    xlim(area_xlim)
    ylim(area_ylim)
    xlabel('x [m]')
    ylabel('y [m]')
    axis('equal')
    axis('tight')
    
    % update the current drawing
    drawnow;
%     pause(0.1)
end


 

