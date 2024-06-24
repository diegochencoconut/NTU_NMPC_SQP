function [grad_cost, grad_state] = find_gradient(input, delta, start_state, ref_states, input_size, state_size, Ts, Q, R, N)

    grad_cost = zeros(1, input_size * N);
    grad_state = zeros(state_size * N, input_size * N);

    for i = 1:N*input_size
        temp_input_plus = input;
        temp_input_plus(i) = temp_input_plus(i) + delta;
        temp_input_minus = input;
        temp_input_minus(i) = temp_input_minus(i) - delta;
        temp_input_plus = reshape(temp_input_plus, input_size, []);
        
        temp_state_plus = model(start_state, Ts, temp_input_plus, N);
        temp_state_minus = model(start_state, Ts, temp_input_minus, N);
        temp_state_diff = (temp_state_plus - temp_state_minus) ./ (2*delta);
        
        temp_cost_plus = cost_function(temp_state_plus-ref_states, temp_input_plus, Q, R, N);
        temp_cost_minus = cost_function(temp_state_minus-ref_states, temp_input_minus, Q, R, N);
        
        grad_cost(i) = (temp_cost_plus - temp_cost_minus) ./ (2*delta);
        grad_state(:, i) = reshape(temp_state_diff, [], 1);
    end

    grad_cost = grad_cost';
    grad_state = grad_state';
end