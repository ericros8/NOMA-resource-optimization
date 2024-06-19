%% Main

% Parameters
params=initParams();

% Result matrices
T_SIC = zeros(params.Nrun, length(params.Np));
T_no_SIC = zeros(params.Nrun, length(params.Np));
T_QL_SIC = zeros(params.Nrun, length(params.Np)); 
T_QL_noSIC = zeros(params.Nrun, length(params.Np));  

% Counters for lost packets
LostPackets_noSIC = zeros(params.Nrun, length(params.Np));
LostPackets_SIC = zeros(params.Nrun, length(params.Np));
LostPackets_QL_SIC = zeros(params.Nrun, length(params.Np));
LostPackets_QL_noSIC = zeros(params.Nrun, length(params.Np));

% Plot user positions
userPositions = plotUserDistribution(params);

rewards = zeros(params.Nep, length(params.Np), params.Nrun);
t_episodesQL = zeros(params.Nep, length(params.Np), params.Nrun);
nw_usedQL = zeros(params.Nep, length(params.Np), params.Nrun);

for npIndex = 1:length(params.Np) 
    
    fprintf('Starting simulation for Np = %d\n', params.Np(npIndex));

    %Packet distribution
    P = initializePacketDistribution(params, npIndex);
    
    %{
    %%%%%% TEST BETA %%%%%%%%
    channelGains = calculateGains(userPositions, params);
    SNR0 = zeros(1,params.Np(npIndex));
    for i = 1:params.Np

        SNR0(i)= (params.Px * channelGains(P(i,1),P(i,2))^2)/params.N0;
    end
    beta = 0.5*min(SNR0);
    params.beta = beta;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %}

    % Simulations
    for run = 1:params.Nrun

        fprintf('Run %d of %d for Np=%d\n', run, params.Nrun, params.Np(npIndex));
        
        %Resource assignment (no SIC)
        [t_noSIC, lost_noSIC] = runResourceAssignment(params, P, userPositions, false);
        T_no_SIC(run, npIndex) = t_noSIC;
        LostPackets_noSIC(run, npIndex) = lost_noSIC;

        %Resource assignment (SIC)
        [t_SIC, lost_SIC] = runResourceAssignment(params, P, userPositions, true);
        T_SIC(run, npIndex) = t_SIC;
        LostPackets_SIC(run, npIndex) = lost_SIC;
        
        % Resource assignment with QL (SIC)
        [Q_SIC, t_QL_SIC_run, lost_QL_SIC_run, episode_rewards,t_episodes, nw_episodes] = runResourceAssignmentWithQL(params, P, userPositions, true);
        T_QL_SIC(run, npIndex) = t_QL_SIC_run;
        LostPackets_QL_SIC(run, npIndex) = lost_QL_SIC_run;
        rewards(:, npIndex, run) = episode_rewards;
        t_episodesQL(:, npIndex, run) = t_episodes;
        nw_usedQL(:, npIndex, run) = nw_episodes;

        % Resource assignment with QL (no SIC)
        [Q_noSIC, t_QL_noSIC_run, lost_QL_noSIC_run, ~, ~, ~] = runResourceAssignmentWithQL(params, P, userPositions, false);
        T_QL_noSIC(run, npIndex) = t_QL_noSIC_run;
        LostPackets_QL_noSIC(run, npIndex) = lost_QL_noSIC_run;

    end
end

% Average slots used
avgSlots_noSIC = mean(T_no_SIC, 1);
avgSlots_SIC = mean(T_SIC, 1);
avgSlots_QL_SIC = mean(T_QL_SIC, 1);
avgSlots_QL_noSIC = mean(T_QL_noSIC, 1);

% Average lost packets 
percLost_noSIC = mean(LostPackets_noSIC ./ repmat(params.Np, params.Nrun, 1) * 100, 1);
percLost_SIC = mean(LostPackets_SIC ./ repmat(params.Np, params.Nrun, 1) * 100, 1);
percLost_QL_SIC = mean(LostPackets_QL_SIC ./ repmat(params.Np, params.Nrun, 1) * 100, 1);
percLost_QL_noSIC = mean(LostPackets_QL_noSIC ./ repmat(params.Np, params.Nrun, 1) * 100, 1);


%% Plot results

figure;
% Subplot for time slots:
subplot(2,1,1);  
hold on;
plot(params.Np, avgSlots_noSIC, '-ob', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'No SIC');
plot(params.Np, avgSlots_SIC, '-or', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'SIC');
plot(params.Np, avgSlots_QL_SIC, '-og', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QL with SIC');
plot(params.Np, avgSlots_QL_noSIC, '-om', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'QL without SIC');
title('Average Number of Time Slots');
xlabel('Number of Packets (Np)');
ylabel('Average Time Slots');
grid on;
legend('show');

% Subplot for lost packets
subplot(2,1,2); 
hold on;
plot(params.Np, percLost_noSIC, '-ob', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Packets Lost No SIC (%)');
plot(params.Np, percLost_SIC, '-or', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Packets Lost SIC (%)');
plot(params.Np, percLost_QL_SIC, '-og', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Packets Lost QL with SIC (%)');
plot(params.Np, percLost_QL_noSIC, '-om', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Packets Lost QL without SIC (%)');
title('Average Percentage of Packets Lost');
xlabel('Number of Packets (Np)');
ylabel('Percentage Lost Packets (%)');
grid on;
legend('show');

hold off;

% Plot of reward

figure;
hold on;
colors = lines(length(params.Np)); 
for npIndex = 1:length(params.Np)
    meanRewards = mean(rewards(:, npIndex, :), 3);  % average of all Nrun
    plot(1:params.Nep, meanRewards, 'LineWidth', 2, 'Color', colors(npIndex, :), 'DisplayName', sprintf('Np = %d', params.Np(npIndex)));
end
xlabel('Episode');
ylabel('Average Reward');
title('Average Reward During Training');
legend show;
grid on;
hold off;

%{
% Plot of reward subplots for each Np
figure;
colors = lines(length(params.Np)); 
for npIndex = 1:length(params.Np)
    subplot(length(params.Np), 1, npIndex); % subplot for each Np
    hold on;
    meanRewards = mean(rewards(:, npIndex, :), 3);
    plot(1:params.Nep, meanRewards, 'LineWidth', 2, 'Color', colors(npIndex, :));
    title(sprintf('Average Reward During Training for Np = %d', params.Np(npIndex)));
    xlabel('Episode');
    ylabel('Average Reward');
    grid on;
    hold off;
end

%legend(arrayfun(@(x) sprintf('Np = %d', x), params.Np, 'UniformOutput', false));
%}

% Plot of Average Time Slots in Training
figure;
hold on;
colors = lines(length(params.Np)); 
for npIndex = 1:length(params.Np)
    meanTimeSlots = mean(t_episodesQL(:, npIndex, :), 3); 
    plot(1:params.Nep, meanTimeSlots, 'LineWidth', 2, 'Color', colors(npIndex, :), 'DisplayName', sprintf('Np = %d', params.Np(npIndex)));
end
xlabel('Episode');
ylabel('Average Time Slots Used');
title('Evolution of Average Time Slots Used During Training');
legend show;
grid on;
hold off;

% Plot of Average WRB in Training
figure;
hold on;
colors = lines(length(params.Np)); 
for npIndex = 1:length(params.Np)
    meanNW = mean(nw_usedQL(:, npIndex, :), 3);  
    plot(1:params.Nep, meanNW, 'LineWidth', 2, 'Color', colors(npIndex, :), 'DisplayName', sprintf('Np = %d', params.Np(npIndex)));
end
xlabel('Episode');
ylabel('Average WRB Used');
title('Evolution of WRB Used During Training');
legend show;
grid on;
hold off;


% Plot epsilon

plotEpsilon(params);


% Plot best action
maxNpIndex = params.Np(5);
maxNwIndex = params.Nw; 

optimalActions = zeros(maxNpIndex, maxNwIndex);

for i = 1:maxNpIndex
    for j = 1:maxNwIndex
        [~, optimalActions(i, j)] = max(Q_SIC(i, j, :));
    end
end

% Best action
[Qmax, bestAction] = max(Q_SIC, [], 3); 
allZeros = all(Q_SIC == 0, 3);

bestAction(allZeros) = 2; % set best action to 2 when all Q values are zero

% Interpolate matrix
[X, Y] = meshgrid(1:5, 1:50);
[Xq, Yq] = meshgrid(1:0.1:5, 1:0.1:50);
bestActionInterp = interp2(X, Y, double(bestAction), Xq, Yq, 'linear');

% Heat Chart
figure;
imagesc(bestActionInterp);
colorbar;
title('Best Action Heatmap (Interpolated)');
xlabel('Number of Resources (Nw)');
ylabel('Number of Packets (Np)');

yticks(0:100:500);
yticklabels(0:10:50); 
xticks(1:10:50);
xticklabels(1:5);

colormap(jet);
caxis([1 3]); 
colorbar('Ticks', [1, 2, 3], 'TickLabels', {'Decrease', 'Maintain', 'Increase'});

%% SNIR

function SNIR = calculateSNIR(h, Px, N0, InTx, P, useSIC)
    
    SNIR = zeros(1, length(InTx));

    for j_R = 1:length(InTx)
        
        I_receptor = P(InTx(j_R), 2);
        interferingSignals = 0;
        P_i_j = Px * h(P(InTx(j_R), 1), I_receptor)^2;
        
        for i_T = 1:length(InTx)             
            if i_T ~= j_R
                P_k_j =  Px * h(P(InTx(i_T), 1), I_receptor)^2;

                if useSIC
                    if P_k_j <= P_i_j
                        interferingSignals = interferingSignals + P_k_j;
                    end
                else
                    interferingSignals = interferingSignals + P_k_j;
                end
            end
        end
        SNIR(j_R) = P_i_j/ (N0 + interferingSignals);
    end
end

function h_dB = calculateChannelLoss(d, referenceLoss, pathLossExponent, referenceDistance, sigmaShadowing, sigmaFading)

    pathLoss = -10 * log10(referenceLoss) - 10 * pathLossExponent * log10(d / referenceDistance);
    
    shadowing = randn * sigmaShadowing;
    
    fading = 10 * log10(abs(randn)^2 * sigmaFading);
    
    h_dB = pathLoss - shadowing - fading;
end

%% Parameters

function params = initParams()
    % Simulation parameters
    params.Nn = 10;  % nodes
    params.areaSize = [100, 100]; % area size (m)
    params.Px = 1; % power (W)
    params.N0 = 1e-5; % noise power
    params.pathLossExponent = 2.5; 
    params.referenceDistance = 1; % (m)
    params.referenceLoss = 1; 
    params.sigmaShadowing = 2;
    params.sigmaFading = 4; 
    params.beta = 0.00000005; % SNIR threshold
    params.Np = [10,20,30,40,50]; % number of packets
    params.Nmax = 5; 
    params.Nrun = 2000; % number of iterations of each simulation

    % QL
    params.Nep = 1000; % number of episodes in training
    params.Ep = 1;
    params.Nw = 5; 
    params.gamma = 0.2; % discount factor
    params.alpha = 0.8; % learning rate
    
    
    params.epsilon_start = 1;
    params.epsilon_end = 0.1;
    params.epsilon_decay = exp(log(params.epsilon_end / params.epsilon_start) / (params.Nep));
    

    params.delta = 0.9; % 0.1, 0.3, 0.9 (cost factor)

end

%% User Distribution

function userPositions = plotUserDistribution(params)
    % Plot and setup user distribution
    figure;
    hold on;
    title(sprintf('User distribution'));
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
    axis([0 params.areaSize(1) 0 params.areaSize(2)]);
    grid on;
    
    userPositions = [rand(params.Nn,1)*params.areaSize(1), rand(params.Nn,1)*params.areaSize(2)];
    
    scatter(userPositions(:,1), userPositions(:,2), 'filled');
    hold off;
end

%% Packet Distribution

function P = initializePacketDistribution(params, npIndex)
    % Packet distribution (ensuring no self-transmissions)
    P = randi(params.Nn, [params.Np(npIndex), 2]);
    for i = 1:params.Np(npIndex)
        while P(i,1) == P(i,2)
            P(i,2) = randi(params.Nn);
        end
    end
end

%% Channel Gains

function channelGains = calculateGains(userPositions, params)

    nodePairs = nchoosek(1:params.Nn,2); % generates all possible node pairs

    % Path loss and channel gains
    h = zeros(params.Nn, params.Nn);
    channelGains = zeros(params.Nn, params.Nn);
   

    for i = 1:size(nodePairs, 1)
        pair = nodePairs(i,:);
        distance = sqrt(sum((userPositions(pair(1),:) - userPositions(pair(2),:)).^2));
        h_dB = calculateChannelLoss(distance, params.referenceLoss, params.pathLossExponent, params.referenceDistance, params.sigmaShadowing, params.sigmaFading);
        
        %line([userPositions(pair(1),1) userPositions(pair(2),1)], ...
        %     [userPositions(pair(1),2) userPositions(pair(2),2)], 'Color', 'r', 'LineWidth', 0.5);
        %midPoint = (userPositions(pair(1),:) + userPositions(pair(2),:)) / 2;
        %text(midPoint(1), midPoint(2), sprintf('%0.2f dB', h_dB),'Color', 'blue', 'FontSize',8);

        h(pair(1), pair(2)) = h_dB;
        h(pair(2), pair(1)) = h_dB;
        
        channelGains(pair(1), pair(2)) = 10^(h_dB/10);
        channelGains(pair(2), pair(1)) = 10^(h_dB/10);
    end
end

%% Resource Assignment SIC & no-SIC

function [t, packetsLost] = runResourceAssignment(params, P_original, userPositions, useSIC)
    
    channelGains = calculateGains(userPositions, params);
    P = [P_original, ones(size(P_original, 1), 1)];  % add transmission status (1 = not sent)
    np = sum(P(:, 3));  % total packets to send
    Tmax = np/3; 
    t = 0;  % time slot 
    packetsLost = 0;

    while np > 0 && t < Tmax
        t = t + 1;

        % Distribute packets across the available single WRB
        IndTx = zeros(1, params.Nmax);  
        ind = 1;
        for k = 1:length(P)
            if P(k, 3) == 1  % if packet not sent
                if ind <= params.Nmax  
                    IndTx(1, ind) = k;
                    ind = ind + 1;
                end
            end
            if ind > params.Nmax
                break;  % exit if resource is full
            end
        end

        % Transmit packets
        IndTx1 = IndTx(IndTx > 0); 
        for aux = 1:length(IndTx1)
            packetIndex = IndTx1(aux);
            SNIR_ij = calculateSNIR(channelGains, params.Px, params.N0, IndTx1, P, useSIC);
        
            if SNIR_ij(aux) > params.beta
                P(packetIndex, 3) = 0;  % packet sent
                np = np - 1;  
            
            end
    
        end
    end
    if np > 0
        packetsLost = np; 
    end
end



%% Resource Assignment with QL (SIC & no-SIC)

function [Q, t_QL, lost_QL,episode_rewards, t_episodes, nw_episodes] = runResourceAssignmentWithQL(params, P_original, userPositions, useSIC)
    maxNpIndex = params.Np(end);
    maxNwIndex = params.Nw;
    Nactions = 3;

    % Initialize Q-table
    Q = zeros(maxNpIndex, maxNwIndex, Nactions);

    alpha = params.alpha;  % learning rate
    gamma = params.gamma;  % discount factor

    t_QL = 0;
    lost_QL = 0;
 
    episode_rewards = zeros(params.Nep, 1);
    t_episodes = zeros(params.Nep, 1);
    nw_episodes = zeros(params.Nep, 1);

    % Run episodes of Q-learning
    for iep = 1:params.Nep
        
        %Epsilon 1
        %epsilon = max(params.epsilon_end, params.epsilon_start * params.epsilon_decay ^ iep);

        %Epsilon 2
        steepness = 10;  
        midpoint = params.Nep / 1.8;  
        epsilon = 1 / (1 + exp(steepness * (iep - midpoint) / params.Nep));

        %Epsilon 3
        %epsilon = params.epsilon_start - (params.epsilon_start - params.epsilon_end) * (iep - 1) / (params.Nep - 1);

        channelGains = calculateGains(userPositions, params);
        P = [P_original, ones(size(P_original, 1), 1)];  % third column for transmission status (1 = not sent)

        % Initial state
        nw = 1; 
        np = sum(P(:, 3));  
        Tmax = np/3; 
        t = 0; 

        while np > 0 && t < Tmax
            t = t + 1;

            npIndex = max(1, min(np, maxNpIndex));
            nwIndex = max(1, min(nw, maxNwIndex));
            
            % Action selection
            if rand < epsilon
                action = randi([-1, 1]);
            else
                [maxQ, actionIndex] = max(Q(npIndex, nwIndex, :));
                if maxQ == 0  % if all Q-values are 0 don't change WRBs
                    action = 0;  
                else
                    action = actionIndex - 2;  
                end            
            end

            % Update number of WRBs used
            nw = max(1, min(nw + action, params.Nw));
            
            IndTx = zeros(nw, params.Nmax);
            ind = 1;
            % Assign unsent packets
            for k = 1:length(P)
                if P(k, 3) == 1  % if packet not sent
                    rowIndex = mod(ind - 1, nw) + 1;
                    colIndex = ceil(ind / nw);
                    if colIndex <= params.Nmax  % if within column limit
                        IndTx(rowIndex, colIndex) = k;
                        ind = ind + 1;
                    end
                end
            end

            % Transmit packets and update states
            for iW = 1:nw
                IndTx1 = IndTx(iW, IndTx(iW, :) > 0);  % remove zeros
                for aux = 1:length(IndTx1)
                    packetIndex = IndTx1(aux);
                    SNIR_ij = calculateSNIR(channelGains, params.Px, params.N0, IndTx1, P, useSIC);
                    if SNIR_ij(aux) > params.beta
                        P(packetIndex, 3) = 0;  % packet sent
                        np = np - 1;
                    end
                end
            end

            % Reward and update Q-table
            reward = -np * params.Ep * sum(params.delta.^(0:(nw - 1)));
            
            if np > 0  
                bestNextAction = max(Q(np, nw, :), [], 'all');
            else
                bestNextAction = 0;
            end

            Q(npIndex, nwIndex, action + 2) = (1 - alpha) * Q(npIndex, nwIndex, action + 2) + alpha * (reward + gamma * bestNextAction);

        end

        % Results for this episode
        episode_rewards(iep, 1) = reward;
        t_episodes(iep) = t;
        nw_episodes(iep) = nw;

        %Final results from the last episode
        if iep == params.Nep
            t_QL = t;
            if np > 0
                lost_QL = np; 
            end        
        end
    end
end

%% Epsilon 1
%{
function plotEpsilon(params)
    epsilons = zeros(params.Nep, 1);
    for i = 1:params.Nep
        epsilons(i) = max(params.epsilon_end, params.epsilon_start * params.epsilon_decay ^ i);
    end

    figure;
    plot(1:params.Nep, epsilons, 'b-', 'LineWidth', 2);
    title('Epsilon over Episodes');
    xlabel('Episodes');
    ylabel('Epsilon');
    grid on;
end
%}

%% Epsilon 2
%%{
function plotEpsilon(params)
    epsilons = zeros(params.Nep, 1);
    steepness = 8;  % Increases the inclination
    midpoint = params.Nep / 3;  % Can be adjusted to initiate decline earlier

    for i = 1:params.Nep
        epsilons(i) = 1 / (1 + exp(steepness * (i - midpoint) / params.Nep));
    end

    figure;
    plot(1:params.Nep, epsilons, 'b-', 'LineWidth', 2);
    title('Epsilon Over Episodes');
    xlabel('Episodes');
    ylabel('Epsilon');
    grid on;
end
%%}
%% Epsilon 3
%{
function plotEpsilon(params)
    epsilons = zeros(params.Nep, 1);
    for i = 1:params.Nep
        % Linear decay from epsilon_start to epsilon_end
        epsilons(i) = params.epsilon_start - (params.epsilon_start - params.epsilon_end) * (i - 1) / (params.Nep - 1);
    end

    figure;
    plot(1:params.Nep, epsilons, 'b-', 'LineWidth', 2);
    title('Epsilon Decay Over Episodes');
    xlabel('Episodes');
    ylabel('Epsilon');
    grid on;
end
%}
