% calculating the cost of current state and current input
% state: the states
% input: the inputs
% Q: the cost matrix of state
% R: the cost matrix of input

function cost = cost_function(state, input, Q, R, N)   
    cost = 0;
    for i = 1:N
        cost = cost + state(:,i)'*Q*state(:,i) + input(:,i)'*R*input(:,i);
    end
end