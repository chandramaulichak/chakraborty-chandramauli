function [A, b, x_hat] = generate_problem_instance(n, p)
    randn('state', 0);
    rand('state', 0);
    
    A = randn(p, n);
    while rank(A) < p
        A = randn(p, n);
    end
    
    x_hat = rand(n, 1);
    b = A * x_hat;
end

function [x, lambda] = standard_newton(A, b, x0, tol, max_iter)
    n = length(x0);
    p = size(A, 1);
    x = x0;
    
    for iter = 1:max_iter
        grad = 1 + log(x);
        H = diag(1 ./ x);
        
        KKT_matrix = [H, A'; A, zeros(p)];
        rhs = [-grad; zeros(p, 1)];
        
        delta = KKT_matrix \ rhs;
        dx = delta(1:n);
        dlambda = delta(n+1:end);
        
        x = x + dx;
        lambda = dlambda;
        
        if norm(dx) < tol
            break;
        end
    end
end

function [x, lambda] = infeasible_newton(A, b, x0, tol, max_iter)
    n = length(x0);
    p = size(A, 1);
    x = x0;
    lambda = zeros(p, 1);
    
    for iter = 1:max_iter
        grad = 1 + log(x);
        H = diag(1 ./ x);
        
        r_dual = grad + A' * lambda;
        r_pri = A * x - b;
        
        KKT_matrix = [H, A'; A, zeros(p)];
        rhs = [-r_dual; -r_pri];
        
        delta = KKT_matrix \ rhs;
        dx = delta(1:n);
        dlambda = delta(n+1:end);
        
        x = x + dx;
        lambda = lambda + dlambda;
        
        if norm([r_dual; r_pri]) < tol
            break;
        end
    end
end


function [x, lambda] = dual_newton(A, b, x0, tol, max_iter)
    p = size(A, 1);
    lambda = zeros(p, 1);
    
    for iter = 1:max_iter
        x = exp(-1 - A' * lambda);
        grad = A * x - b;
        H = -A * diag(x) * A';
        
        dlambda = H \ (-grad);
        lambda = lambda + dlambda;
        
        if norm(grad) < tol
            break;
        end
    end
end

function [x1, lambda1, x2, lambda2, x3, lambda3] = hwk7(x0)
    n = 100;
    p = 30;
    [A, b, x_hat] = generate_problem_instance(n, p);
    
    tol = 1e-6;
    max_iter = 100;
    
    [x1, lambda1] = standard_newton(A, b, x_hat, tol, max_iter);
    [x2, lambda2] = infeasible_newton(A, b, x_hat, tol, max_iter);
    [x3, lambda3] = dual_newton(A, b, x0, tol, max_iter);
end

x0 = ones(100, 1);


[x1, lambda1, x2, lambda2, x3, lambda3] = hwk7(x0);

disp('Standard Newton Method:');
disp('x1:'); disp(x1);
disp('lambda1:'); disp(lambda1);

disp('Infeasible Start Newton Method:');
disp('x2:'); disp(x2);
disp('lambda2:'); disp(lambda2);

disp('Dual Newton Method:');
disp('x3:'); disp(x3);
disp('lambda3:'); disp(lambda3);
