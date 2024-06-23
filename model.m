% Calculating the states with initial state and input values
% init_state: the initial state
% Ts: sampling time
% input: a series of input
% N: Predicting steps
% states: states after the model, from i = 1
function states = model(init_state, Ts, input, N)
   
    states = zeros(length(init_state), N+1);
    states(:,1) = init_state;
    for i = 2:N+1
        states(:,i) = model_equation(states(:,i-1), Ts, input(:,i-1));
    end
    states = states(:,2:end);
end


function next_state = model_equation(current_state, Ts, input)
    g = 9.8;
    kdm = 1;
    A_CT = [0 0 1 0 0 0;
            0 0 0 1 0 0;
            0 0 -kdm 0 g 0;
            0 0 0 -kdm 0 -g;
            0 0 0 0 0 0;
            0 0 0 0 0 0];
    B_CT = [0 0;
            0 0;
            0 0;
            0 0;
            1 0;
            0 1];
    next_state = current_state + Ts * (A_CT * current_state + B_CT * input);

    % kd = 1;
    % m = 1;
    % g = 9.8;

    % next_state(1) = current_state(1) + Ts * current_state(3);
    % next_state(2) = current_state(2) + Ts * current_state(4);
    % next_state(2) = 0;
    % next_state(3) = current_state(3) + Ts * (-kd/m*current_state(3) + g*tan(current_state(5)));
    % next_state(4) = current_state(4) + Ts * (-kd/m*current_state(4) - g*tan(current_state(6))/cos(current_state(5)));
    % next_state(4) = 0;
    % next_state(5) = current_state(5) + Ts * input(1);
    % next_state(6) = current_state(6) + Ts * input(2);
    % next_state(6) = 0;
end