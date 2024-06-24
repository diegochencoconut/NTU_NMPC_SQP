close all
clear; clc;

%% Simulation Conditions
Ts = 0.1;
real_Ts = 0.01;
N = 10;
simTime = 10;
Time = Ts:Ts:simTime;
options = optimoptions('quadprog','Algorithm','active-set','Display','off');
MAXIMUM_LOOP = 20;

%% Initialize
state_size = 6;         % Check consistent with model.m
input_size = 2;         % Check consistent with model.m
init_state = [1.9,5.9,-0.538,-0.548,-0.2,-0.2]';   % Check consistent
ref_state = [-0.91,4.12,0,0,0,0]' .* ones(state_size, (length(Time)+N));

best_state = zeros(state_size, length(Time));
best_input = zeros(input_size, length(Time));

input = zeros(input_size, N);

%% Optimization parameters
delta = 0.01;
start_state = init_state;
Q = diag([1 1 1e-4 1e-4 1e-4 1e-4]);
R = diag([1e-2 1e-2]);
[grad_cost, grad_state] = find_gradient(input, delta, start_state, ref_state(:, 1:N), input_size, state_size, Ts, Q, R, N);
state = model(start_state, Ts, input, N);

%% Simulation core

for timeTick = 1:length(Time)
    Time(timeTick)
    J = eye(input_size * N);
    H = inv(J);
    
    % SQP core
    for loop = 1:MAXIMUM_LOOP

        % Constraints
        max_input = 4*ones(input_size*N, 1);
        min_input = -4*ones(input_size*N, 1);
        
        max_state = diag([3.5, 7.5,2,2,0.2,0.2]);
        min_state = diag([-1, -4.5,-2,-2,-0.2,-0.2]);
        max_state = reshape(max_state * ones(state_size, N), [], 1);
        min_state = reshape(min_state * ones(state_size, N), [], 1);

        % Find best direction
        oneline_input = reshape(input, [], 1);
        oneline_state = reshape(state, [], 1);
        % du = quadprog(H, grad_cost',[eye(input_size*N); -eye(input_size*N); grad_state'; -grad_state'],[max_input-oneline_input; oneline_input-min_input; max_state-oneline_state ; oneline_state-min_state],[],[],[],[],input,options);
        du = quadprog(H, grad_cost',[eye(input_size*N); -eye(input_size*N); grad_state'; -grad_state'],[max_input-oneline_input; oneline_input-min_input; max_state-oneline_state ; oneline_state-min_state],[],[],[],[],input,options);
        
        % update value
        input = input + reshape(du, input_size, N);
        state = model(start_state, Ts, input, N);

        % store current gradient
        grad_cost_old = grad_cost;

        % find next gradient
        [grad_cost, grad_state] = find_gradient(input, delta, start_state, ref_state(:, timeTick:timeTick+(N-1)), input_size, state_size, Ts, Q, R, N);

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
    if (loop == MAXIMUM_LOOP)
        disp("MAXIMUM LOOP REACHED");
    end

    % real model (with faster Ts)
    best_input(:,timeTick) = input(:, 1);
    real_plant_input = input(:, 1) .* ones(input_size, Ts/real_Ts);
    real_plant_state = model(start_state, real_Ts, real_plant_input, Ts/real_Ts);
    best_state(:, timeTick) = real_plant_state(:, end);
    % best_state(:,timeTick) = state(:, 1);
    start_state = real_plant_state(:, end);

    % calculate the init of next iteration
    input = [input(:, 2:end), input(:, end)];
    state = model(start_state, Ts, input, N);

    [grad_cost, grad_state] = find_gradient(input, delta, start_state, ref_state(:, timeTick+1:timeTick+N), input_size, state_size, Ts, Q, R, N);
end
%% Plot results
best_state_plot = [init_state best_state];
best_input_plot = [best_input, best_input(:,end-1)];
Time_plot = [0 Time];
figure;
subplot(4, 2, 1);
plot(Time_plot, best_state_plot(1, :));
ylabel('$x$[m]', 'Interpreter','latex');
subplot(4, 2, 2);
plot(Time_plot, best_state_plot(2, :));
ylabel('$y$[m]', 'Interpreter','latex');
subplot(4, 2, 3);
plot(Time_plot, best_state_plot(3, :));
ylabel('$\dot{x}$[m]', 'Interpreter','latex');
subplot(4, 2, 4);
plot(Time_plot, best_state_plot(4, :));
ylabel('$\dot{y}$[m]', 'Interpreter','latex');
subplot(4, 2, 5);
plot(Time_plot, best_state_plot(5, :));
ylabel('$\theta$[rad]', 'Interpreter','latex');
subplot(4, 2, 6);
plot(Time_plot, best_state_plot(6, :));
ylabel('$\psi$[rad]', 'Interpreter','latex');
subplot(4, 2, 7);
plot(Time_plot, best_input_plot(1, :));
ylabel('$\dot{\theta}$[rad/s]', 'Interpreter','latex');
xlabel('Time[s]', 'Interpreter','latex');
subplot(4, 2, 8);
plot(Time_plot, best_input_plot(2, :));
ylabel('$\dot{\psi}$[rad/s]', 'Interpreter','latex');
xlabel('Time[s]', 'Interpreter','latex');