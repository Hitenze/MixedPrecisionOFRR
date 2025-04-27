close all;
clear;
clc;

% Load test configurations
configs = tests.test4.test_configs();

% Select test configuration (default: configuration 3 for bcsstk15)
test_idx = 1; 

% Get current test parameters
params = configs.params{test_idx};

% Use parameters
matrix_name = params.matrix_name;
matrix_path = params.matrix_path;
k = params.k;
kdim = params.kdim;
maxiter = params.maxiter;
m = kdim; % Set m = kdim for backward compatibility with any other code that might use m

% Output current test information
fprintf('Running sparse eigenvalue residual test configuration %d: %s\n', test_idx, params.description);
fprintf('Parameters: matrix=%s, k=%d, kdim=%d, maxiter=%d\n', ...
        matrix_name, k, kdim, maxiter);

% Load the test matrix
fprintf('Loading sparse matrix from %s\n', matrix_path);
load(matrix_path);
% scale the matrix to avoid overflow
if strcmp(matrix_name, 'bcsstk01')
   K = Problem.A / 100000000;
elseif strcmp(matrix_name, 'bcsstk03')
   K = Problem.A / 1000000000;
elseif strcmp(matrix_name, '1138_bus')
   K = Problem.A / 100;
else
   error('Unsupported matrix: %s', matrix_name);
end
[n1, n2] = size(K);
fprintf('Matrix size: %d x %d with %d non-zeros\n', n1, n2, nnz(K));

% Ensure matrix is symmetric positive definite (SPD)
K = (K + K')/2;  % Ensure symmetry
min_eig = min(eig(full(K)));
if min_eig <= 0
   fprintf('Warning: Matrix not SPD, adding diagonal shift to ensure positive eigenvalues\n');
   K = K + abs(min_eig)*1.01*speye(size(K));
end

% Convert to CSR format
K_csr = src.csr.sparse2csr(K);

stable_tol = 0e-10;
stable_tol_half = 1e-10;
seed = 42;
rng(seed);

tic;
[eig_double_mgs, vec_double_mgs, k_eig_double_mgs] = test_arnoldi_mgs_eig(K_csr, k, kdim, maxiter, seed, ...
                                                                         @double, @double, ...
                                                                         @double);
toc;

tic;
[eig_single_mgs, vec_single_mgs, k_eig_single_mgs] = test_arnoldi_mgs_eig(K_csr, k, kdim, maxiter, seed, ...
                                                                         @single, @single, ...
                                                                         @single);
toc;

tic;
[eig_half_mgs, vec_half_mgs, k_eig_half_mgs] = test_arnoldi_mgs_eig(K_csr, k, kdim, maxiter, seed, ...
                                                                   @single, @half, ...
                                                                   @double);
toc;

tic;
[eig_double_mgs_gev, vec_double_mgs_gev, k_eig_double_mgs_gev] = test_arnoldi_mgs_eig_gev(K_csr, k, kdim, maxiter, seed, stable_tol, ...
                                                                                         @double, @double, ...
                                                                                         @double);
toc;

tic;
[eig_single_mgs_gev, vec_single_mgs_gev, k_eig_single_mgs_gev] = test_arnoldi_mgs_eig_gev(K_csr, k, kdim, maxiter, seed, stable_tol, ...
                                                                                         @single, @single, ...
                                                                                         @double);
toc;

tic;
[eig_half_mgs_gev, vec_half_mgs_gev, k_eig_half_mgs_gev] = test_arnoldi_mgs_eig_gev(K_csr, k, kdim, maxiter, seed, stable_tol_half, ...
                                                                                   @single, @half, ...
                                                                                   @double);
toc;

tic;
[eig_double_hess_gev, vec_double_hess_gev, k_eig_double_hess_gev] = test_arnoldi_hess_eig(K_csr, k, kdim, maxiter, seed, stable_tol, ...
                                                                                         @double, @double, ...
                                                                                         @double);
toc;

tic;
[eig_single_hess_gev, vec_single_hess_gev, k_eig_single_hess_gev] = test_arnoldi_hess_eig(K_csr, k, kdim, maxiter, seed, stable_tol, ...
                                                                                         @single, @single, ...
                                                                                         @double);
toc;

tic;
[eig_half_hess_gev, vec_half_hess_gev, k_eig_half_hess_gev] = test_arnoldi_hess_eig(K_csr, k, kdim, maxiter, seed, stable_tol_half, ...
                                                                                   @single, @half, ...
                                                                                   @double);
toc;

k_new = min([k_eig_double_mgs, k_eig_single_mgs, k_eig_half_mgs, ...
             k_eig_double_mgs_gev, k_eig_single_mgs_gev, k_eig_half_mgs_gev, ...
             k_eig_double_hess_gev, k_eig_single_hess_gev, k_eig_half_hess_gev]);
if k_new ~= k
   warning('insufficient k obtained, %d/%d', k_new, k);
end

% Calculate residual norms for each eigenpair
res_double_mgs = calculate_residual_norms(K_csr, vec_double_mgs, eig_double_mgs, k_eig_double_mgs);
res_single_mgs = calculate_residual_norms(K_csr, vec_single_mgs, eig_single_mgs, k_eig_single_mgs);
res_half_mgs = calculate_residual_norms(K_csr, vec_half_mgs, eig_half_mgs, k_eig_half_mgs);

res_double_mgs_gev = calculate_residual_norms(K_csr, vec_double_mgs_gev, eig_double_mgs_gev, k_eig_double_mgs_gev);
res_single_mgs_gev = calculate_residual_norms(K_csr, vec_single_mgs_gev, eig_single_mgs_gev, k_eig_single_mgs_gev);
res_half_mgs_gev = calculate_residual_norms(K_csr, vec_half_mgs_gev, eig_half_mgs_gev, k_eig_half_mgs_gev);

res_double_hess_gev = calculate_residual_norms(K_csr, vec_double_hess_gev, eig_double_hess_gev, k_eig_double_hess_gev);
res_single_hess_gev = calculate_residual_norms(K_csr, vec_single_hess_gev, eig_single_hess_gev, k_eig_single_hess_gev);
res_half_hess_gev = calculate_residual_norms(K_csr, vec_half_hess_gev, eig_half_hess_gev, k_eig_half_hess_gev);

tic;
fprintf('Computing true eigenvalues (this may take a while)...\n');
[eig_vectors, eig_values] = eigs(K, k, 'largestabs','SubspaceDimension',kdim);
eig_true = sort(diag(eig_values), 'descend');
eig_true = eig_true(1:k);
toc;

%% Plotting

fig = figure(1);
clf;

% Set line width for better visibility
fig_width = 800;
fig_height = 300;
lineWidth = 3;
marker_size = 10;
label_font_size = 20;
title_font_size = 25;

colors = src.utils.get_colors();

% Plotting with specified colors and line styles
semilogy(1:k_eig_double_mgs, res_double_mgs, ...
        'LineWidth', lineWidth, 'Color', colors.myblue);
hold on;
semilogy(1:k_eig_single_mgs, res_single_mgs, ...
        'LineWidth', lineWidth, 'Color', colors.myblue, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(res_single_mgs), 'MarkerSize', marker_size);
semilogy(1:k_eig_half_mgs, res_half_mgs, ...
        'LineWidth', lineWidth, 'Color', colors.myblue, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(res_half_mgs), 'MarkerSize', marker_size);
semilogy(1:k_eig_double_mgs_gev, res_double_mgs_gev, ...
        'LineWidth', lineWidth, 'Color', colors.myred);
semilogy(1:k_eig_single_mgs_gev, res_single_mgs_gev, ...
        'LineWidth', lineWidth, 'Color', colors.myred, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(res_single_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_eig_half_mgs_gev, res_half_mgs_gev, ...
        'LineWidth', lineWidth, 'Color', colors.myred, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(res_half_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_eig_double_hess_gev, res_double_hess_gev, ...
        'LineWidth', lineWidth, 'Color', colors.mypurple);
semilogy(1:k_eig_single_hess_gev, res_single_hess_gev, ...
        'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(res_single_hess_gev), 'MarkerSize', marker_size);
semilogy(1:k_eig_half_hess_gev, res_half_hess_gev, ...
        'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(res_half_hess_gev), 'MarkerSize', marker_size);

% Only show title for first test configuration
if test_idx == 1
   title('Residual Norms ||Av - λv|| of Approximated Eigenpairs', 'FontSize', title_font_size);
end
xlabel('Eigenvalue index', 'FontSize', label_font_size);
ylabel('Residual norm', 'FontSize', label_font_size);

% No legend here - we use separate legend file
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100, 100, fig_width, fig_height];

% Include test configuration number and matrix name in the filename
saveas(fig, sprintf('test_40_%s_config%d.png', matrix_name, test_idx));

fig = figure(2);
clf;

% Set line width for better visibility
fig_width = 800;
fig_height = 300;
lineWidth = 3;
label_font_size = 20;
title_font_size = 25;

% Use mygreen for eigenvalues
semilogy(eig_true, 'LineWidth', lineWidth, 'Color', colors.mygreen);
hold on;
semilogy([1,k], [1,1]*eps('double'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '--');
semilogy([1,k], [1,1]*eps('single'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '-.');
semilogy([1,k], [1,1]*eps('half'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', ':');

% Only show title for first test configuration
if test_idx == 1
   title('True eigenvalues and machine epsilon', 'FontSize', title_font_size);
end
xlabel('Eigenvalue index', 'FontSize', label_font_size);
ylabel('Eigenvalue', 'FontSize', label_font_size);

% No legend here - we use separate legend file
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100+fig_width, 100, fig_width, fig_height];

% Include test configuration number and matrix name in the filename
saveas(fig, sprintf('test_40_%s_eig_config%d.png', matrix_name, test_idx));

% Save data to log file
log_file = sprintf('test40_%s_config%d_data.log', matrix_name, test_idx);
fid = fopen(log_file, 'w');

% Write header
fprintf(fid, 'Eigenvalue_Index\t');
fprintf(fid, 'True_Eigenvalue\t');
fprintf(fid, 'MGS_double\tMGS_single\tMGS_half\t');
fprintf(fid, 'MGS_GEV_double\tMGS_GEV_single\tMGS_GEV_half\t');
fprintf(fid, 'Hess_GEV_double\tHess_GEV_single\tHess_GEV_half\n');

% Write data
for i = 1:k
   fprintf(fid, '%d\t', i);
   fprintf(fid, '%.15e\t', eig_true(i));
   
   % MGS residuals
   if i <= k_eig_double_mgs
      if res_double_mgs(i) > eps('double')
         fprintf(fid, '%.15e\t', res_double_mgs(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_single_mgs
      if res_single_mgs(i) > eps('double')
         fprintf(fid, '%.15e\t', res_single_mgs(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_half_mgs
      if res_half_mgs(i) > eps('double')
         fprintf(fid, '%.15e\t', res_half_mgs(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   
   % MGS GEV residuals
   if i <= k_eig_double_mgs_gev
      if res_double_mgs_gev(i) > eps('double')
         fprintf(fid, '%.15e\t', res_double_mgs_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_single_mgs_gev
      if res_single_mgs_gev(i) > eps('double')
         fprintf(fid, '%.15e\t', res_single_mgs_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_half_mgs_gev
      if res_half_mgs_gev(i) > eps('double')
         fprintf(fid, '%.15e\t', res_half_mgs_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   
   % Hessenberg GEV residuals
   if i <= k_eig_double_hess_gev
      if res_double_hess_gev(i) > eps('double')
         fprintf(fid, '%.15e\t', res_double_hess_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_single_hess_gev
      if res_single_hess_gev(i) > eps('double')
         fprintf(fid, '%.15e\t', res_single_hess_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\t', eps('double'));
      end
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_eig_half_hess_gev
      if res_half_hess_gev(i) > eps('double')
         fprintf(fid, '%.15e\n', res_half_hess_gev(i));
      else
         % avoid zero in semilogy
         fprintf(fid, '%.15e\n', eps('double'));
      end
   else
      fprintf(fid, 'NaN\n');
   end
end

fclose(fid);
fprintf('Data saved to %s\n', log_file);

% Function to calculate residual norms ||Av - λv||
function residuals = calculate_residual_norms(A, V, lambda, k)
   residuals = zeros(k, 1);
   for i = 1:k
      v = double(V(:, i));
      lambda_i = double(lambda(i));
      % Calculate residual in double precision using CSR matrix-vector product
      Av = src.csr.csrmv(A, 'N', v, 'precision_compute', @double, 'precision_output', @double);
      res = norm(Av - lambda_i * v, 2);
      residuals(i) = res / lambda_i;
   end
end

function [eig_values, eig_vectors, k] = test_arnoldi_mgs_eig(K, k, kdim, maxiter, seed, ...
                                                            precision_compute, precision_output, ...
                                                            precision_eig)

   src.krylov.arnoldi_mgs_eig([], [], [], 'precision_compute', precision_compute, ...
                             'precision_output', precision_output, ...
                             'precision_A_matvec_compute', precision_compute, ...
                             'precision_A_matvec_output', precision_output, ...
                             'precision_eigenvalues', precision_eig, ...
                             'seed', seed, ...
                             'maxiter', maxiter, ...
                             'kdim', kdim, ...
                             'generalized', false, ...
                             'matvec', @src.csr.csrmv, ...
                             'check_params', true);
   
   % Run Arnoldi MGS method
   [V, D] = src.krylov.arnoldi_mgs_eig(K, K.nrows, k, 'precision_compute', precision_compute, ...
                                      'precision_output', precision_output, ...
                                      'precision_A_matvec_compute', precision_compute, ...
                                      'precision_A_matvec_output', precision_output, ...
                                      'precision_eigenvalues', precision_eig, ...
                                      'seed', seed, ...
                                      'maxiter', maxiter, ...
                                      'kdim', kdim, ...
                                      'generalized', false, ...
                                      'matvec', @src.csr.csrmv);
   
   % Extract and sort eigenvalues and eigenvectors
   [eig_values, idx] = sort(diag(D), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
   eig_vectors = V(:, idx(1:k));
end

function [eig_values, eig_vectors, k] = test_arnoldi_mgs_eig_gev(K, k, kdim, maxiter, seed, stable_tol, ...
                                                                precision_compute, precision_output, ...
                                                                precision_eig)

   src.krylov.arnoldi_mgs_eig([], [], [], 'precision_compute', precision_compute, ...
                             'precision_output', precision_output, ...
                             'precision_A_matvec_compute', precision_compute, ...
                             'precision_A_matvec_output', precision_output, ...
                             'precision_eigenvalues', precision_eig, ...
                             'seed', seed, ...
                             'maxiter', maxiter, ...
                             'kdim', kdim, ...
                             'generalized', true, ...
                             'generalized_tol', stable_tol, ...
                             'matvec', @src.csr.csrmv, ...
                             'check_params', true);
   
   [V, D] = src.krylov.arnoldi_mgs_eig(K, K.nrows, k, 'precision_compute', precision_compute, ...
                                      'precision_output', precision_output, ...
                                      'precision_A_matvec_compute', precision_compute, ...
                                      'precision_A_matvec_output', precision_output, ...
                                      'precision_eigenvalues', precision_eig, ...
                                      'seed', seed, ...
                                      'maxiter', maxiter, ...
                                      'kdim', kdim, ...
                                      'generalized', true, ...
                                      'generalized_tol', stable_tol, ...
                                      'matvec', @src.csr.csrmv);
   
   % Extract and sort eigenvalues and eigenvectors
   [eig_values, idx] = sort(diag(D), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
   eig_vectors = V(:, idx(1:k));
end

function [eig_values, eig_vectors, k] = test_arnoldi_hess_eig(K, k, kdim, maxiter, seed, stable_tol, ...
                                                             precision_compute, precision_output, ...
                                                             precision_eig)

   src.krylov.arnoldi_hess_eig([], [], [], 'precision_compute', precision_compute, ...
                              'precision_output', precision_output, ...
                              'precision_A_matvec_compute', precision_compute, ...
                              'precision_A_matvec_output', precision_output, ...
                              'precision_eigenvalues', precision_eig, ...
                              'seed', seed, ...
                              'maxiter', maxiter, ...
                              'kdim', kdim, ...
                              'generalized_tol', stable_tol, ...
                              'matvec', @src.csr.csrmv, ...
                              'check_params', true);
   
   [V, D] = src.krylov.arnoldi_hess_eig(K, K.nrows, k, 'precision_compute', precision_compute, ...
                                       'precision_output', precision_output, ...
                                       'precision_A_matvec_compute', precision_compute, ...
                                       'precision_A_matvec_output', precision_output, ...
                                       'precision_eigenvalues', precision_eig, ...
                                       'seed', seed, ...
                                       'maxiter', maxiter, ...
                                       'kdim', kdim, ...
                                       'generalized_tol', stable_tol, ...
                                       'matvec', @src.csr.csrmv);
   
   [eig_values, idx] = sort(diag(D), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
   eig_vectors = V(:, idx(1:k));
end 