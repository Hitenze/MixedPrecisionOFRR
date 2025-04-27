close all;
clear;
clc;

% Load test configurations
configs = tests.test3.test_configs();

% Select test configuration (default: configuration 1)
test_idx = 1; 

% Get current test parameters
params = configs.params{test_idx};

% Use parameters
n1 = params.n1;
n2 = params.n2;
l = params.l;
f = params.f;
s = params.s;
m = params.m;
k = params.k;
iter = params.iter;
maxiter = params.maxiter;

% Output current test information
fprintf('Running SVD test configuration %d: %s\n', test_idx, params.description);
fprintf('Parameters: n1=%d, n2=%d, l=%g, s=%g, m=%d, k=%d, iter=%d, maxiter=%d\n', ...
        n1, n2, l, s, m, k, iter, maxiter);

reorth_tol = 1/sqrt(2); 
stable_tol = 0e-10;
seed = 42;
rng(seed);
X = src.kernel.generate_pts(n1,20);
Y = X(randsample(n1,n2),:);
K = src.kernel.gaussian(X,Y,l,f,s);

tic;
[svd_double_mgs, k_svd_double_mgs] = test_subspace_iter_svd_mgs(K, m, k, maxiter, iter, seed, ...
                                                                @double, @double, ...
                                                                @double, @double, ...
                                                                @double, @double, ...
                                                                @double);
toc;

tic;
[svd_single_mgs, k_svd_single_mgs] = test_subspace_iter_svd_mgs(K, m, k, maxiter, iter, seed, ...
                                                                @single, @single, ...
                                                                @single, @single, ...
                                                                @single, @single, ...
                                                                @double);
toc;

tic;
[svd_half_mgs, k_svd_half_mgs] = test_subspace_iter_svd_mgs(K, m, k, maxiter, iter, seed, ...
                                                                @single, @half, ...
                                                                @single, @half, ...
                                                                @single, @single, ...
                                                                @double);
toc;

tic;
[svd_double_mgs_gev, k_svd_double_mgs_gev] = test_subspace_iter_svd_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @double, @double, ...
                                                                            @double, @double, ...
                                                                            @double, @double, ...
                                                                            @double);
toc;

tic;
[svd_single_mgs_gev, k_svd_single_mgs_gev] = test_subspace_iter_svd_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @single, @single, ...
                                                                            @single, @single, ...
                                                                            @single, @single, ...
                                                                            @double);
toc;

tic;
[svd_half_mgs_gev, k_svd_half_mgs_gev] = test_subspace_iter_svd_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @single, @half, ...
                                                                            @single, @half, ...
                                                                            @single, @single, ...
                                                                            @double);
toc;

tic;
[svd_double_hess_gev, k_svd_double_hess_gev] = test_subspace_iter_svd_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @double, @double, ...
                                                                            @double, @double, ...
                                                                            @double, @double, ...
                                                                            @double);
toc;

tic;
[svd_single_hess_gev, k_svd_single_hess_gev] = test_subspace_iter_svd_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @single, @single, ...
                                                                            @single, @single, ...
                                                                            @single, @single, ...
                                                                            @double);
toc;

tic;
[svd_half_hess_gev, k_svd_half_hess_gev] = test_subspace_iter_svd_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                                            @single, @half, ...
                                                                            @single, @half, ...
                                                                            @single, @single, ...
                                                                            @double);
toc;

k_new = min([k_svd_double_mgs, k_svd_single_mgs, k_svd_half_mgs, ...
             k_svd_double_mgs_gev, k_svd_single_mgs_gev, k_svd_half_mgs_gev, ...
             k_svd_double_hess_gev, k_svd_single_hess_gev, k_svd_half_hess_gev]);
if k_new ~= k
   warning('insufficient k obtained, %d/%d', k_new, k);
end

tic;
svd_true = sort(svd(K), 'descend');
svd_true = svd_true(1:k);
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
semilogy(1:k_svd_double_mgs, max(abs(svd_double_mgs(1:k_svd_double_mgs) - svd_true(1:k_svd_double_mgs))./abs(svd_true(1:k_svd_double_mgs)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myblue);
hold on;
semilogy(1:k_svd_single_mgs, max(abs(svd_single_mgs(1:k_svd_single_mgs) - svd_true(1:k_svd_single_mgs))./abs(svd_true(1:k_svd_single_mgs)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myblue, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(svd_single_mgs), 'MarkerSize', marker_size);
semilogy(1:k_svd_half_mgs, max(abs(svd_half_mgs(1:k_svd_half_mgs) - svd_true(1:k_svd_half_mgs))./abs(svd_true(1:k_svd_half_mgs)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myblue, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(svd_half_mgs), 'MarkerSize', marker_size);
semilogy(1:k_svd_double_mgs_gev, max(abs(svd_double_mgs_gev(1:k_svd_double_mgs_gev) - svd_true(1:k_svd_double_mgs_gev))./abs(svd_true(1:k_svd_double_mgs_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myred);
semilogy(1:k_svd_single_mgs_gev, max(abs(svd_single_mgs_gev(1:k_svd_single_mgs_gev) - svd_true(1:k_svd_single_mgs_gev))./abs(svd_true(1:k_svd_single_mgs_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myred, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(svd_single_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_svd_half_mgs_gev, max(abs(svd_half_mgs_gev(1:k_svd_half_mgs_gev) - svd_true(1:k_svd_half_mgs_gev))./abs(svd_true(1:k_svd_half_mgs_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.myred, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(svd_half_mgs_gev), 'MarkerSize', marker_size);
semilogy(1:k_svd_double_hess_gev, max(abs(svd_double_hess_gev(1:k_svd_double_hess_gev) - svd_true(1:k_svd_double_hess_gev))./abs(svd_true(1:k_svd_double_hess_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.mypurple);
semilogy(1:k_svd_single_hess_gev, max(abs(svd_single_hess_gev(1:k_svd_single_hess_gev) - svd_true(1:k_svd_single_hess_gev))./abs(svd_true(1:k_svd_single_hess_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '--', 'Marker', '^', 'MarkerIndices', 1:3:length(svd_single_hess_gev), 'MarkerSize', marker_size);
semilogy(1:k_svd_half_hess_gev, max(abs(svd_half_hess_gev(1:k_svd_half_hess_gev) - svd_true(1:k_svd_half_hess_gev))./abs(svd_true(1:k_svd_half_hess_gev)), eps('double')), ...
        'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '-.', 'Marker', 'o', 'MarkerIndices', 1:1:length(svd_half_hess_gev), 'MarkerSize', marker_size);

% Only show title for first test configuration
if test_idx == 1
   title('Relative Error of Approximated Singular Values', 'FontSize', title_font_size);
end
xlabel('Singular value index', 'FontSize', label_font_size);
ylabel('Relative error', 'FontSize', label_font_size);

% No legend here - we use separate legend file
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100, 100, fig_width, fig_height];

% Include test configuration number in the filename
saveas(fig, sprintf('test_30_config%d.png', test_idx));

fig = figure(2);
clf;

% Set line width for better visibility
fig_width = 800;
fig_height = 300;
lineWidth = 3;
label_font_size = 20;
title_font_size = 25;

% Use mygreen for singular values
semilogy(svd_true, 'LineWidth', lineWidth, 'Color', colors.mygreen);
hold on;
semilogy([1,k], [1,1]*eps('double'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '--');
semilogy([1,k], [1,1]*eps('single'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', '-.');
semilogy([1,k], [1,1]*eps('half'), 'LineWidth', lineWidth, 'Color', 'black', 'LineStyle', ':');

% Only show title for first test configuration
if test_idx == 1
   title('True singular values and machine epsilon', 'FontSize', title_font_size);
end
xlabel('Singular value index', 'FontSize', label_font_size);
ylabel('Singular value', 'FontSize', label_font_size);

% No legend here - we use separate legend file
% legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100+fig_width, 100, fig_width, fig_height];

% Include test configuration number in the filename
saveas(fig, sprintf('test_30_svd_config%d.png', test_idx));

% Save data to log file
log_file = sprintf('test30_config%d_data.log', test_idx);
fid = fopen(log_file, 'w');

% Write header
fprintf(fid, 'SingularValue_Index\t');
fprintf(fid, 'True_SingularValue\t');
fprintf(fid, 'MGS_double\tMGS_single\tMGS_half\t');
fprintf(fid, 'MGS_GEV_double\tMGS_GEV_single\tMGS_GEV_half\t');
fprintf(fid, 'Hess_GEV_double\tHess_GEV_single\tHess_GEV_half\n');

% Write data
for i = 1:k
   fprintf(fid, '%d\t', i);
   fprintf(fid, '%.15e\t', svd_true(i));
   
   % MGS errors
   if i <= k_svd_double_mgs
      fprintf(fid, '%.15e\t', abs(svd_double_mgs(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_single_mgs
      fprintf(fid, '%.15e\t', abs(svd_single_mgs(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_half_mgs
      fprintf(fid, '%.15e\t', abs(svd_half_mgs(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   
   % MGS GEV errors
   if i <= k_svd_double_mgs_gev
      fprintf(fid, '%.15e\t', abs(svd_double_mgs_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_single_mgs_gev
      fprintf(fid, '%.15e\t', abs(svd_single_mgs_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_half_mgs_gev
      fprintf(fid, '%.15e\t', abs(svd_half_mgs_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   
   % Hessenberg GEV errors
   if i <= k_svd_double_hess_gev
      fprintf(fid, '%.15e\t', abs(svd_double_hess_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_single_hess_gev
      fprintf(fid, '%.15e\t', abs(svd_single_hess_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\t');
   end
   if i <= k_svd_half_hess_gev
      fprintf(fid, '%.15e\n', abs(svd_half_hess_gev(i) - svd_true(i))/abs(svd_true(i)));
   else
      fprintf(fid, 'NaN\n');
   end
end

fclose(fid);
fprintf('Data saved to %s\n', log_file);

function [svd_values, k] = test_subspace_iter_svd_mgs(K, m, k, maxiter, iter, seed, ...
                                                     precision_subspace_compute, precision_subspace_output, ...
                                                     precision_mgs_compute, precision_mgs_output, ...
                                                     precision_rr_compute, precision_rr_output, ...
                                                     precision_eig)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, ...
              'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, ...
                                  'precision_output', precision_mgs_output);
   
   src.rr.svd_rr([], [], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, 'check_params', true);
   rr_function = @(A, U, V) src.rr.svd_rr(A, U, V, 'precision_matvec_compute', precision_rr_compute, ...
                                         'precision_matvec_output', precision_rr_output, ...
                                         'precision_A_matvec_compute', precision_subspace_compute, ...
                                         'precision_A_matvec_output', precision_subspace_output, ...
                                         'precision_eig', precision_eig);
   
   src.subspace.subspace_iter_svd([], [], [], [], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [nrows, ncols] = size(K);
   [~, S, ~] = src.subspace.subspace_iter_svd(K, nrows, ncols, m, 'maxiter', maxiter, 'step_size', iter, ...
                                            'precision_compute', precision_subspace_compute, ...
                                            'precision_output', precision_subspace_output, ...
                                            'seed', seed, 'function_orth', orth_function, ...
                                            'function_rr', rr_function);
   
   svd_values = sort(diag(S), 'descend');
   k = min(k, length(svd_values));
   svd_values = svd_values(1:k);
end

function [svd_values, k] = test_subspace_iter_svd_mgs_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                         precision_subspace_compute, precision_subspace_output, ...
                                                         precision_mgs_compute, precision_mgs_output, ...
                                                         precision_rr_compute, precision_rr_output, ...
                                                         precision_eig)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, ...
              'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, ...
                                  'precision_output', precision_mgs_output);
   
   src.rr.svd_rr([], [], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, 'use_generalized', true, ...
                'generalized_tol', stable_tol, 'check_params', true);
   rr_function = @(A, U, V) src.rr.svd_rr(A, U, V, 'precision_matvec_compute', precision_rr_compute, ...
                                         'precision_matvec_output', precision_rr_output, ...
                                         'precision_A_matvec_compute', precision_subspace_compute, ...
                                         'precision_A_matvec_output', precision_subspace_output, ...
                                         'precision_eig', precision_eig, 'use_generalized', true, ...
                                         'generalized_tol', stable_tol);
   
   src.subspace.subspace_iter_svd([], [], [], [], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [nrows, ncols] = size(K);
   [~, S, ~] = src.subspace.subspace_iter_svd(K, nrows, ncols, m, 'maxiter', maxiter, 'step_size', iter, ...
                                            'precision_compute', precision_subspace_compute, ...
                                            'precision_output', precision_subspace_output, ...
                                            'seed', seed, 'function_orth', orth_function, ...
                                            'function_rr', rr_function);
   
   svd_values = sort(diag(S), 'descend');
   k = min(k, length(svd_values));
   svd_values = svd_values(1:k);
end

function [svd_values, k] = test_subspace_iter_svd_hess_gev(K, m, k, maxiter, iter, seed, stable_tol, ...
                                                          precision_subspace_compute, precision_subspace_output, ...
                                                          precision_hess_compute, precision_hess_output, ...
                                                          precision_rr_compute, precision_rr_output, ...
                                                          precision_eig)

   src.qr.hessenberg([], 'precision_compute', precision_hess_compute, ...
                    'precision_output', precision_hess_output, 'check_params', true);
   orth_function = @(A) src.qr.hessenberg(A, 'precision_compute', precision_hess_compute, ...
                                        'precision_output', precision_hess_output);
   
   src.rr.svd_rr([], [], [], 'precision_matvec_compute', precision_rr_compute, ...
                'precision_matvec_output', precision_rr_output, ...
                'precision_A_matvec_compute', precision_subspace_compute, ...
                'precision_A_matvec_output', precision_subspace_output, ...
                'precision_eig', precision_eig, 'use_generalized', true, ...
                'generalized_tol', stable_tol, 'check_params', true);
   rr_function = @(A, U, V) src.rr.svd_rr(A, U, V, 'precision_matvec_compute', precision_rr_compute, ...
                                         'precision_matvec_output', precision_rr_output, ...
                                         'precision_A_matvec_compute', precision_subspace_compute, ...
                                         'precision_A_matvec_output', precision_subspace_output, ...
                                         'precision_eig', precision_eig, 'use_generalized', true, ...
                                         'generalized_tol', stable_tol);
   
   src.subspace.subspace_iter_svd([], [], [], [], 'maxiter', maxiter, 'step_size', iter, ...
                                'precision_compute', precision_subspace_compute, ...
                                'precision_output', precision_subspace_output, ...
                                'seed', seed, 'function_orth', orth_function, ...
                                'function_rr', rr_function, 'check_params', true);
   
   [nrows, ncols] = size(K);
   [~, S, ~] = src.subspace.subspace_iter_svd(K, nrows, ncols, m, 'maxiter', maxiter, 'step_size', iter, ...
                                            'precision_compute', precision_subspace_compute, ...
                                            'precision_output', precision_subspace_output, ...
                                            'seed', seed, 'function_orth', orth_function, ...
                                            'function_rr', rr_function);
   
   svd_values = sort(diag(S), 'descend');
   k = min(k, length(svd_values));
   svd_values = svd_values(1:k);
end