close all;
clear;
clc;

% Load test configurations
configs = tests.test2.test_configs();

% Select test configuration
test_idx = 1; 

% Get current test parameters
params = configs.params{test_idx};

% Use parameters
n = params.n;
l = params.l;
f = params.f;
s = params.s;
m = params.m;
k = params.k;
iter = params.iter;
maxiter = params.maxiter;

% Output current test information
fprintf('Running test configuration %d: %s\n', test_idx, params.description);
fprintf('Parameters: n=%d, l=%g, s=%g, m=%d, k=%d, iter=%d, maxiter=%d\n', ...
        n, l, s, m, k, iter, maxiter);

reorth_tol = 1/sqrt(2);
stable_tol = 0e-10;
seed = 42;
rng(seed);
X = src.kernel.generate_pts(n,20);
K = src.kernel.gaussian(X,X,l,f,s);

% subspace, qr, rr, eig

tic;
[eig_double_mgs, k_double_mgs] = test_subspace_iter_eig_mgs(K, m, k, maxiter, iter, seed, ...
                                                            @double, @double, ...
                                                            @double, @double, ...
                                                            @double, @double, ...
                                                            @double);
toc;

tic;
[eig_single_mgs, k_single_mgs] = test_subspace_iter_eig_mgs(K, m, k, maxiter, iter, seed, ...
                                                            @single, @single, ...
                                                            @single, @single, ...
                                                            @single, @single, ...
                                                            @double);
toc;

tic;
[eig_half_mgs, k_half_mgs] = test_subspace_iter_eig_mgs(K, m, k, maxiter, iter, seed, ...
                                                            @single, @half, ...
                                                            @single, @half, ...
                                                            @single, @single, ...
                                                            @double);
toc;

tic;
[eig_double_mgs_gev, k_double_mgs_gev] = test_subspace_iter_eig_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                       @double, @double, ...
                                                                       @double, @double, ...
                                                                       @double, @double, ...
                                                                       @double);
toc;

tic;
[eig_single_mgs_gev, k_single_mgs_gev] = test_subspace_iter_eig_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                       @single, @single, ...
                                                                       @single, @single, ...
                                                                       @single, @single, ...
                                                                       @double);
toc;

tic;
[eig_half_mgs_gev, k_half_mgs_gev] = test_subspace_iter_eig_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                   @single, @half, ...
                                                                   @single, @half, ...
                                                                   @single, @single, ...
                                                                   @double);
toc;

tic;
[eig_double_hess_gev, k_double_hess_gev] = test_subspace_iter_eig_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                           @double, @double, ...
                                                                           @double, @double, ...
                                                                           @double, @double, ...
                                                                           @double);
toc;

tic;
[eig_single_hess_gev, k_single_hess_gev] = test_subspace_iter_eig_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                           @single, @single, ...
                                                                           @single, @single, ...
                                                                           @single, @single, ...
                                                                           @double);
toc;

tic;
[eig_half_hess_gev, k_half_hess_gev] = test_subspace_iter_eig_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                        @single, @half, ...
                                                                        @single, @half, ...
                                                                        @single, @single, ...
                                                                        @double);
toc;

k_new = min([k_double_mgs, k_single_mgs, k_half_mgs, ...
             k_double_mgs_gev, k_single_mgs_gev, k_half_mgs_gev, ...
             k_double_hess_gev, k_single_hess_gev, k_half_hess_gev]);
if k_new ~= k
   warning('insufficient k obtained, %d/%d', k_new, k);
end

tic;
eig_true = sort(real(eig(K)), 'descend');
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
legend_font_size = 20;
label_font_size = 20;
title_font_size = 25;

colors = src.utils.get_colors();

% Plotting with specified colors and line styles
semilogy(1:k_double_mgs, abs(eig_double_mgs(1:k_double_mgs) - eig_true(1:k_double_mgs))./abs(eig_true(1:k_double_mgs)), 'DisplayName', 'double-qr', 'LineWidth', lineWidth, ...
        'Color', colors.myblue);
hold on;
semilogy(1:k_single_mgs, abs(eig_single_mgs(1:k_single_mgs) - eig_true(1:k_single_mgs))./abs(eig_true(1:k_single_mgs)), 'DisplayName', 'single-qr', 'LineWidth', lineWidth, ...
        'Color', colors.myblue, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(eig_single_mgs), 'MarkerSize', marker_size);
semilogy(1:k_half_mgs, abs(eig_half_mgs(1:k_half_mgs) - eig_true(1:k_half_mgs))./abs(eig_true(1:k_half_mgs)), 'DisplayName', 'half-qr', 'LineWidth', lineWidth, ...
        'Color', colors.myblue, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(eig_half_mgs), 'MarkerSize', marker_size);
semilogy(1:k_double_mgs_gev, abs(eig_double_mgs_gev(1:k_double_mgs_gev) - eig_true(1:k_double_mgs_gev))./abs(eig_true(1:k_double_mgs_gev)), 'DisplayName', 'double-qr-gev', 'LineWidth', lineWidth, ...
        'Color', colors.myred);
semilogy(1:k_single_mgs_gev, abs(eig_single_mgs_gev(1:k_single_mgs_gev) - eig_true(1:k_single_mgs_gev))./abs(eig_true(1:k_single_mgs_gev)), 'DisplayName', 'single-qr-gev', 'LineWidth', lineWidth, ...
        'Color', colors.myred, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(eig_single_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_half_mgs_gev, abs(eig_half_mgs_gev(1:k_half_mgs_gev) - eig_true(1:k_half_mgs_gev))./abs(eig_true(1:k_half_mgs_gev)), 'DisplayName', 'half-qr-gev', 'LineWidth', lineWidth, ...
        'Color', colors.myred, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(eig_half_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_double_hess_gev, abs(eig_double_hess_gev(1:k_double_hess_gev) - eig_true(1:k_double_hess_gev))./abs(eig_true(1:k_double_hess_gev)), 'DisplayName', 'double-hess-gev', 'LineWidth', lineWidth, ...
        'Color', colors.mypurple);
semilogy(1:k_single_hess_gev, abs(eig_single_hess_gev(1:k_single_hess_gev) - eig_true(1:k_single_hess_gev))./abs(eig_true(1:k_single_hess_gev)), 'DisplayName', 'single-hess-gev', 'LineWidth', lineWidth, ...
        'Color', colors.mypurple, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(eig_single_hess_gev), 'MarkerSize', marker_size);
semilogy(1:k_half_hess_gev, abs(eig_half_hess_gev(1:k_half_hess_gev) - eig_true(1:k_half_hess_gev))./abs(eig_true(1:k_half_hess_gev)), 'DisplayName', 'half-hess-gev', 'LineWidth', lineWidth, ...
        'Color', colors.mypurple, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(eig_half_hess_gev), 'MarkerSize', marker_size);

if test_idx == 1
   title('Relative Error of Approximated Eigenvalues', 'FontSize', title_font_size);
end
xlabel('Eigenvalue index', 'FontSize', label_font_size);
ylabel('Relative error', 'FontSize', label_font_size);

% Increase legend font size
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100, 100, fig_width, fig_height];

% Include test configuration number in the filename
saveas(fig, sprintf('test_20_config%d.png', test_idx));

fig = figure(2);
clf;

% Set line width for better visibility
fig_width = 800;
fig_height = 300;
lineWidth = 3;
legend_font_size = 20;
label_font_size = 20;
title_font_size = 25;

semilogy(eig_true, 'DisplayName', 'Eigenvalues', 'LineWidth', lineWidth, 'Color', colors.mygreen);
hold on;
semilogy([1,k], [1,1]*eps('double'), 'DisplayName', 'Machine epsilon (double)', 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '--');
semilogy([1,k], [1,1]*eps('single'), 'DisplayName', 'Machine epsilon (single)', 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '-.');
semilogy([1,k], [1,1]*eps('half'), 'DisplayName', 'Machine epsilon (half)', 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', ':');

% Include test configuration number in the title
if test_idx == 1
   title('True eigenvalues and machine epsilon', 'FontSize', title_font_size);
end
xlabel('Eigenvalue index', 'FontSize', label_font_size);
ylabel('Eigenvalue', 'FontSize', label_font_size);

% Increase legend font size
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100+fig_width, 100, fig_width, fig_height];

% Include test configuration number in the filename
saveas(fig, sprintf('test_20_eig_config%d.png', test_idx));

% Save data to log file
log_file = sprintf('test20_config%d_data.log', test_idx);
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
   
   % MGS errors
   if i <= k_double_mgs
      fprintf(fid, '%.15e\t', abs(eig_double_mgs(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_single_mgs
      fprintf(fid, '%.15e\t', abs(eig_single_mgs(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_half_mgs
      fprintf(fid, '%.15e\t', abs(eig_half_mgs(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   
   % MGS GEV errors
   if i <= k_double_mgs_gev
      fprintf(fid, '%.15e\t', abs(eig_double_mgs_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_single_mgs_gev
      fprintf(fid, '%.15e\t', abs(eig_single_mgs_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_half_mgs_gev
      fprintf(fid, '%.15e\t', abs(eig_half_mgs_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   
   % Hessenberg GEV errors
   if i <= k_double_hess_gev
      fprintf(fid, '%.15e\t', abs(eig_double_hess_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_single_hess_gev
      fprintf(fid, '%.15e\t', abs(eig_single_hess_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_half_hess_gev
      fprintf(fid, '%.15e\n', abs(eig_half_hess_gev(i) - eig_true(i))/abs(eig_true(i)));
   else
      fprintf(fid, 'NaN\n');
   end
end

fclose(fid);
fprintf('Data saved to %s\n', log_file);

return;

function [eig_values, k] = test_subspace_iter_eig_mgs(K, m, k, maxiter, iter, seed, ...
                                                     precision_subspace_compute, precision_subspace_output, ...
                                                     precision_mgs_compute, precision_mgs_output, ...
                                                     precision_rr_compute, precision_rr_output, ...
                                                     precision_eig)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, ...
              'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, ...
                                  'precision_output', precision_mgs_output);
   
   src.rr.eig_rr([], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, 'check_params', true);
   rr_function = @(A, V) src.rr.eig_rr(A, V, 'precision_matvec_compute', precision_rr_compute, ...
                                      'precision_matvec_output', precision_rr_output, ...
                                      'precision_A_matvec_compute', precision_subspace_compute, ...
                                      'precision_A_matvec_output', precision_subspace_output, ...
                                      'precision_eig', precision_eig);
   
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [~, D] = src.subspace.subspace_iter_eig(K, size(K, 1), m, 'maxiter', maxiter, 'step_size', iter, ...
                                         'precision_compute', precision_subspace_compute, ...
                                         'precision_output', precision_subspace_output, ...
                                         'seed', seed, 'function_orth', orth_function, ...
                                         'function_rr', rr_function);
   
   eig_values = sort(real(diag(D)), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
end

function [eig_values, k] = test_subspace_iter_eig_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                         precision_subspace_compute, precision_subspace_output, ...
                                                         precision_mgs_compute, precision_mgs_output, ...
                                                         precision_rr_compute, precision_rr_output, ...
                                                         precision_eig)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, ...
              'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, ...
                                  'precision_output', precision_mgs_output);
   
   src.rr.eig_rr([], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, ...
                'use_generalized', true, 'generalized_tol', stable_tol, 'check_params', true);
   rr_function = @(A, V) src.rr.eig_rr(A, V, 'precision_matvec_compute', precision_rr_compute, ...
                                      'precision_matvec_output', precision_rr_output, ...
                                      'precision_A_matvec_compute', precision_subspace_compute, ...
                                      'precision_A_matvec_output', precision_subspace_output, ...
                                      'precision_eig', precision_eig, ...
                                      'use_generalized', true, 'generalized_tol', stable_tol);
   
   src.subspace.subspace_iter_eig([], [],[], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [~, D] = src.subspace.subspace_iter_eig(K, size(K, 1), m, 'maxiter', maxiter, 'step_size', iter, ...
                                         'precision_compute', precision_subspace_compute, ...
                                         'precision_output', precision_subspace_output, ...
                                         'seed', seed, 'function_orth', orth_function, ...
                                         'function_rr', rr_function);
   
   eig_values = sort(real(diag(D)), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
end

function [eig_values, k] = test_subspace_iter_eig_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                          precision_subspace_compute, precision_subspace_output, ...
                                                          precision_hess_compute, precision_hess_output, ...
                                                          precision_rr_compute, precision_rr_output, ...
                                                          precision_eig)

   src.qr.hessenberg([], 'precision_compute', precision_hess_compute, ...
                    'precision_output', precision_hess_output, 'check_params', true);
   
   orth_function = @(A) src.qr.hessenberg(A, 'precision_compute', precision_hess_compute, ...
                                        'precision_output', precision_hess_output);
   
   src.rr.eig_rr([], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, ...
                'use_generalized', true, 'generalized_tol', stable_tol, 'check_params', true);
   rr_function = @(A, V) src.rr.eig_rr(A, V, 'precision_matvec_compute', precision_rr_compute, ...
                                      'precision_matvec_output', precision_rr_output, ...
                                      'precision_A_matvec_compute', precision_subspace_compute, ...
                                      'precision_A_matvec_output', precision_subspace_output, ...
                                      'precision_eig', precision_eig, ...
                                      'use_generalized', true, 'generalized_tol', stable_tol);
   
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [~, D] = src.subspace.subspace_iter_eig(K, size(K, 1), m, 'maxiter', maxiter, 'step_size', iter, ...
                                         'precision_compute', precision_subspace_compute, ...
                                         'precision_output', precision_subspace_output, ...
                                         'seed', seed, 'function_orth', orth_function, ...
                                         'function_rr', rr_function);
   
   eig_values = sort(real(diag(D)), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);
end