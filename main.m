close all
clear; clc;

%% Simulation Conditions
Ts = 0.1;
real_Ts = 0.01;
N = 10;
simTime = 5;
Time = 0:Ts:simTime;
options =  optimset('Display','off');
MAXIMUM_LOOP = 30;

%% Initialize
state_size = 6;         % Check consistent with model.m
input_size = 2;         % Check consistent with model.m
init_state = [1.9,5.9,-0.538,-0.548,-0.2,-0.2]';   % Check consistent

best_state = zeros(state_size, length(Time));
best_input = zeros(input_size, length(Time));

input = zeros(input_size, N);

%% Optimization parameters
delta = 0.01;
start_state = init_state;
Q = diag([1 1 1e-4 1e-4 1e-4 1e-4]);
R = diag([1e-2 1e-2]);
[grad_cost, grad_state] = find_gradient(input, delta, start_state, input_size, state_size, Ts, Q, R, N);
state = model(start_state, Ts, input, N);

%% Simulation core

for timeTick = 1:(length(Time)-1)
    Time(timeTick)
    J = eye(input_size * N);
    H = inv(J);
    
    % SQP core
    for loop = 1:MAXIMUM_LOOP

        % Constraints
        max_input = 4*ones(input_size*N, 1);
        min_input = -4*ones(input_size*N, 1);
        
        % max_state = reshape([Inf 0;0 Inf] * ones(state_size, N), [], 1);
        % min_state = reshape([-Inf 0;0 -Inf]' * ones(state_size, N), [], 1);

        % Find best direction
        oneline_input = reshape(input, [], 1);
        oneline_state = reshape(state, [], 1);
        % du = quadprog(H, grad_cost',[eye(input_size*N); -eye(input_size*N); grad_state'; -grad_state'],[max_input-oneline_input; oneline_input-min_input; max_state-oneline_state ; oneline_state-min_state],[],[],[],[],input,options);
        du = quadprog(H, grad_cost',[eye(input_size*N); -eye(input_size*N)],[max_input-oneline_input; oneline_input-min_input],[],[],[],[],input,options);
        
        % update value
        input = input + reshape(du, input_size, N);
        state = model(start_state, Ts, input, N);

        % store current gradient
        grad_cost_old = grad_cost;

        % find next gradient
        [grad_cost, grad_state] = find_gradient(input, delta, start_state, input_size, state_size, Ts, Q, R, N);

        % BFGS Hessian Approximation
        s = du;
        y = grad_cost - grad_cost_old;
        rho = (1/(s'*y));

        % check convergence
        if (norm(y)/norm(grad_cost) < 1e-3 || norm(du) < 1e-3)
            break
        end
        
        J = (eye(input_size * N) - rho*s*y')*J*(eye(input_size * N) - rho*y*s') + rho*(s*s');
        % J = (H+H')/2;
        H = inv(J);
        H = (H+H')/2;       % to make sure it is hermitian


    end
    loop

    % real model (with faster Ts)
    best_input(:,timeTick) = input(:, 1);
    real_plant_input = input(:, 1) .* ones(input_size, Ts/real_Ts);
    real_plant_state = model(start_state, real_Ts, real_plant_input, Ts/real_Ts);
    best_state(:, timeTick) = real_plant_state(:, end);
    % best_state(:,timeTick) = state(:, 1);
    start_state = real_plant_state(:, end);

    input = [input(:, 2:end), input(:, end)];
    state = model(start_state, Ts, input, N);

    [grad_cost, grad_state] = find_gradient(input, delta, start_state, input_size, state_size, Ts, Q, R, N);
end
%% Plot results
figure;
subplot(2, 1, 1);
plot(Time, [init_state best_state(:,1:end-1)])
subplot(2, 1, 2)
plot(Time, [best_input(:,1:end-1), best_input(:,end-1)])