%% Final Project
% Part 2
% -----------------------------------------------------
% Identify Possible Periodic Orbits around one of 
% the Artificial Equilibriafor a Stationary Solar Sail

clear; clc; close all;

% USER INPUTS

mu = 3.003489e-6;   % Sun-Earth mass ratio

% Artificial equilibrium (From part 1)
x_eq = 1.0090418181;
y_eq = 0.0;
z_eq = 0.0;

% Solar sail parameters
beta  = 0.01;              % Lightness number
n_hat = [1; 0; 0];         % Sail normal direction (unit vector, fixed)

% State at equilibrium (stationary in rotating frame)
Xeq = [x_eq; y_eq; z_eq; 0; 0; 0];

% 1. LINEARIZATION AT ARTIFICIAL EQUILIBRIUM

A = numerical_jacobian(@(X) cr3bp_sail_eom(X, mu, beta, n_hat), Xeq);

[V, D] = eig(A);
eigvals = diag(D);

disp('Eigenvalues at artificial equilibrium:');
disp(eigvals);

% Identify oscillatory modes (complex eigenvalues)
tol = 1e-6;
osc_modes = find(abs(imag(eigvals)) > tol);

disp('Indices of oscillatory modes (imaginary eigenvalues):');
disp(osc_modes);

% 2. BUILD INITIAL GUESS FOR A PERIODIC ORBIT

% Pick one oscillatory mode (e.g., first one)
mode_idx = osc_modes(1);

% Normalize eigenvector and take real part (direction in phase space)
v_mode = V(:, mode_idx);
v_mode = v_mode / norm(v_mode);

epsilon = 1e-4;   % small amplitude
X0_guess = Xeq + epsilon * real(v_mode);

disp('Initial guess for periodic orbit (state):');
disp(X0_guess.');

% 3. REFINE TO A PERIODIC ORBIT VIA SHOOTING

% Unknowns: [x0, y0, z0, vx0, vy0, vz0, T]
% We start near the equilibrium, so use X0_guess and a rough period guess
T_guess = 5;   % rough guess for period (nondimensional)

u0 = [X0_guess; T_guess];

options = optimoptions('fsolve', ...
    'Display','iter', ...
    'TolFun',1e-12, ...
    'TolX',1e-12, ...
    'MaxIterations',400, ...
    'MaxFunctionEvaluations',5000);

u_sol = fsolve(@(u) periodic_shooting_residual(u, mu, beta, n_hat), u0, options);

X0_orbit = u_sol(1:6);
T_orbit  = u_sol(7);

disp('Refined periodic orbit initial condition:');
disp(X0_orbit.');
disp(['Refined period: ', num2str(T_orbit)]);

% 4. INTEGRATE AND PLOT THE PERIODIC ORBIT

options = odeset('Events', @section_event, 'RelTol',1e-12, 'AbsTol',1e-12);

[t, X] = ode45(@(t,X) cr3bp_sail_eom(X, mu, beta, n_hat), [0 50], X0_orbit, options);


figure;
plot3(X(:,1), X(:,2), X(:,3), 'b','LineWidth',1.5); hold on;
plot3(x_eq, y_eq, z_eq, 'ro', 'MarkerSize',8, 'LineWidth',2);
grid on;
xlabel('x [AU]');
ylabel('y [AU]');
zlabel('z [AU]');
title('Periodic orbit around artificial equilibrium (solar sail)');
legend('Periodic orbit','Artificial equilibrium');

% 5. COMPUTE MONODROMY MATRIX AND CLASSIFY STABILITY

% Integrate variational equations + state
M0 = eye(6);                 % initial state transition matrix
X0_aug = [X0_orbit; M0(:)];  % augment state with STM

% Integrate over one period
[t_aug, X_aug] = ode45(@(t,X) variational_eom(t, X, mu, beta, n_hat), [0 T_orbit], X0_aug);

% Extract final STM
Xf_aug = X_aug(end,:)';
M = reshape(Xf_aug(7:end), 6, 6);

% Eigenvalues of monodromy matrix
[evecM, evalM] = eig(M);
mults = diag(evalM);

disp('Floquet multipliers (eigenvalues of monodromy matrix):');
disp(mults);

% CLASSIFICATION
num_real_pairs = 0;
num_complex_pairs = 0;

for k = 1:2:6
    lam1 = mults(k);
    lam2 = mults(k+1);

    if abs(imag(lam1)) < 1e-6 && abs(imag(lam2)) < 1e-6
        num_real_pairs = num_real_pairs + 1;
    else
        num_complex_pairs = num_complex_pairs + 1;
    end
end

if num_real_pairs == 1 && num_complex_pairs == 1
    disp('Stability classification: centre × saddle');
elseif num_real_pairs == 2
    disp('Stability classification: saddle × saddle');
elseif num_complex_pairs == 2
    disp('Stability classification: centre × centre');
else
    disp('Stability classification: mixed/degenerate');
end


% ===== FUNCTIONS =====

function dXdt = cr3bp_sail_eom(X, mu, beta, n_hat)
    % CR3BP with simple solar sail acceleration (fixed n_hat)
    x  = X(1); y  = X(2); z  = X(3);
    vx = X(4); vy = X(5); vz = X(6);

    r1 = sqrt((x + mu)^2       + y^2 + z^2);        % Sun
    r2 = sqrt((x - (1-mu))^2   + y^2 + z^2);        % Earth

    % Gravitational potential derivatives (standard CR3BP)
    Ux = x - (1-mu)*(x+mu)/r1^3 - mu*(x-(1-mu))/r2^3;
    Uy = y - (1-mu)*y/r1^3      - mu*y/r2^3;
    Uz =    -(1-mu)*z/r1^3      - mu*z/r2^3;

    % Simple sail acceleration: constant direction n_hat, magnitude ~ beta/r1^2
    a_sail = beta * ( (1-mu) / r1^2 ) * n_hat;

    ax = 2*vy + Ux + a_sail(1);
    ay = -2*vx + Uy + a_sail(2);
    az = Uz + a_sail(3);

    dXdt = [vx; vy; vz; ax; ay; az];
end

function A = numerical_jacobian(f, X0)
    n = length(X0);
    A = zeros(n);
    h = 1e-6;

    for i = 1:n
        dX    = zeros(n,1);
        dX(i) = h;
        A(:,i) = (f(X0 + dX) - f(X0 - dX)) / (2*h);
    end
end

function R = periodic_shooting_residual(u, mu, beta, n_hat)
    % u = [x0, y0, z0, vx0, vy0, vz0, T]
    X0 = u(1:6);
    T  = u(7);

    % Integrate over one period
    [~, X] = ode45(@(t,X) cr3bp_sail_eom(X, mu, beta, n_hat), [0 T], X0);
    Xf = X(end,:).';

    % Residual: require X(T) = X(0) (closed orbit)
    R = Xf - X0;
end

function dXdt = variational_eom(t, X_aug, mu, beta, n_hat)
    % Extract state and STM
    X = X_aug(1:6);
    M = reshape(X_aug(7:end), 6, 6);

    % Compute dynamics and Jacobian
    f = cr3bp_sail_eom(X, mu, beta, n_hat);
    A = numerical_jacobian(@(X) cr3bp_sail_eom(X, mu, beta, n_hat), X);

    % Variational equation: dM/dt = A*M
    dMdt = A * M;

    % Pack derivative
    dXdt = [f; dMdt(:)];
end

function [value, isterminal, direction] = section_event(t, X)
    % Stop when y = 0 and dy/dt > 0
    value = X(2);        % y = 0
    isterminal = 1;      % stop integration
    direction = +1;      % only detect upward crossings (dy/dt > 0)
end
