function [a, b, convergence_ratio] = newton_logistic(a0, b0, epsilon)
    % Load data
    hwk6p6data;

    % Initial guess
    a = a0;
    b = b0;

    % Parameters for backtracking line search
    alpha = 0.20;
    beta = 0.90;

    % Newton's method
    max_iter = 100;
    convergence_ratio = []; % To store the convergence ratio

    for iter = 1:max_iter
        % Compute gradient
        grad_a = -sum(u(ind_true)) + sum(u .* (exp(a * u + b) ./ (1 + exp(a * u + b))));
        grad_b = -q + sum(exp(a * u + b) ./ (1 + exp(a * u + b)));
        grad = [grad_a; grad_b];

        % Compute Hessian
        H_aa = sum(u.^2 .* (exp(a * u + b) ./ (1 + exp(a * u + b)).^2));
        H_ab = sum(u .* (exp(a * u + b) ./ (1 + exp(a * u + b)).^2));
        H_bb = sum((exp(a * u + b) ./ (1 + exp(a * u + b)).^2));
        Hessian = [H_aa, H_ab; H_ab, H_bb];

        % Newton step
        delta = -Hessian \ grad;
        delta_a = delta(1);
        delta_b = delta(2);

        % Decrement
        lambda_sq = -grad' * delta;

        % Stopping criterion
        if lambda_sq / 2 <= epsilon
            break;
        end

        % Backtracking line search
        t = 1;
        while true
            a_new = a + t * delta_a;
            b_new = b + t * delta_b;

            f_new = -sum(a_new * u(ind_true) + b_new) + sum(log(1 + exp(a_new * u + b_new)));
            f_old = -sum(a * u(ind_true) + b) + sum(log(1 + exp(a * u + b)));

            if f_new <= f_old + alpha * t * (grad' * delta)
                break;
            end
            t = beta * t;
        end

        % Update a and b
        a = a_new;
        b = b_new;

        % Compute convergence ratio for the last few iterates
        if iter > 1
            grad_norm_new = norm(grad);
            grad_norm_old = norm(grad_prev);
            convergence_ratio = [convergence_ratio; log(grad_norm_new) / log(grad_norm_old)];
        end

        % Store current gradient for the next iteration
        grad_prev = grad;
    end
end

hwk6p6data;

% Initial guess
a0 = 5;
b0 = 5;

% Solve for m = 100
m = 100;
[a, b,convergence_ratio] = newton_logistic(a0, b0, 1e-20);
fprintf('Solution for m = 100: a = %.4f, b = %.4f\n', a, b);
fprintf('The last three convergence_ratios are: (%.4f, %.4f, %.4f)',convergence_ratio(end-2:end))
% end(-2::end)
