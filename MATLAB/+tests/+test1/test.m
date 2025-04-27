close all;
clear;
clc;

n = 1000;
ls = 0:0.025:2;
ls = 10.^ls;
s = 1e-01;
k = 30;
iter = 4;
maxiter = 1;
reorth_tol = 1/sqrt(2); 
%reorth_tol = -1;
seed = 815;
rng(seed);
X = src.kernel.generate_pts(n,20);
V = rand(n,k);

ntests = length(ls);
conds_double_X = zeros(ntests,1);
conds_single_X = zeros(ntests,1);
conds_half_X = zeros(ntests,1);
conds_double_cgs = zeros(ntests,1);
conds_single_cgs = zeros(ntests,1);
conds_half_cgs = zeros(ntests,1);
conds_double_mgs = zeros(ntests,1);
conds_single_mgs = zeros(ntests,1);
conds_half_mgs = zeros(ntests,1);
conds_double_hessenberg = zeros(ntests,1);
conds_single_hessenberg = zeros(ntests,1);
conds_half_hessenberg = zeros(ntests,1);

ks_double_X = zeros(ntests,1);
ks_single_X = zeros(ntests,1);
ks_half_X = zeros(ntests,1);
ks_double_cgs = zeros(ntests,1);
ks_single_cgs = zeros(ntests,1);
ks_half_cgs = zeros(ntests,1);
ks_double_mgs = zeros(ntests,1);
ks_single_mgs = zeros(ntests,1);
ks_half_mgs = zeros(ntests,1);
ks_double_hessenberg = zeros(ntests,1);
ks_single_hessenberg = zeros(ntests,1);
ks_half_hessenberg = zeros(ntests,1);

for i = 1:ntests

   l = ls(i);
   K = src.kernel.gaussian(X,X,l,1,s);


   tic;
   [conds_double_X(i), ks_double_X(i)] = test_orth_X(K, k, maxiter, iter, seed, ...
                                                               @double, @double, @double, @double);
   toc;

   tic;
   [conds_single_X(i), ks_single_X(i)] = test_orth_X(K, k, maxiter, iter, seed, ...
                                                               @single, @single, @single, @single);
   toc;

   tic;
   [conds_half_X(i), ks_half_X(i)] = test_orth_X(K, k, maxiter, iter, seed, ...
                                                               @single, @half, @single, @half);
   toc;

   tic;
   [conds_double_cgs(i), ks_double_cgs(i)] = test_orth_cgs(K, k, maxiter, iter, seed, ...
                                                               @double, @double, @double, @double);
   toc;

   tic;
   [conds_single_cgs(i), ks_single_cgs(i)] = test_orth_cgs(K, k, maxiter, iter, seed, ...
                                                               @single, @single, @single, @single);
   toc;

   tic;
   %[conds_half_cgs(i), ks_half_cgs(i)] = test_orth_cgs(K, k, maxiter, iter, seed, ...
   %                                                            @single, @half, @single, @half);
   % avoid different default dropping for this case for a fair comparison with MGS
   [conds_half_cgs(i), ks_half_cgs(i)] = test_orth_cgs_custom_select_tol(K, k, maxiter, iter, seed, ...
                                                               @half, @half, @half, @half, 0);
   toc;

   tic;
   [conds_double_mgs(i), ks_double_mgs(i)] = test_orth_mgs(K, k, maxiter, iter, seed, ...
                                                               @double, @double, @double, @double);
   toc;

   tic;
   [conds_single_mgs(i), ks_single_mgs(i)] = test_orth_mgs(K, k, maxiter, iter, seed, ...
                                                               @single, @single, @single, @single);
   toc;

   tic;
   % avoid different default dropping for this case for a fair comparison with Hessenberg
   [conds_half_mgs(i), ks_half_mgs(i)] = test_orth_mgs_custom_select_tol(K, k, maxiter, iter, seed, ...
                                                                         @half, @half, @half, @half, 0);
   toc;

   tic;
   [conds_double_hessenberg(i), ks_double_hessenberg(i)] = test_orth_hessenberg(K, k, maxiter, iter, seed, ...
                                                                                 @double, @double, @double, @double);
   toc;

   tic;
   [conds_single_hessenberg(i), ks_single_hessenberg(i)] = test_orth_hessenberg(K, k, maxiter, iter, seed, ...
                                                                                 @single, @single, @single, @single);
   toc;
   
   tic;
   % avoid different default dropping for this case for a fair comparison with MGS
   [conds_half_hessenberg(i), ks_half_hessenberg(i)] = test_orth_hessenberg_custom_select_tol(K, k, maxiter, iter, seed, ...
                                                                                 @half, @half, @half, @half, 0);
   toc;

end

%% Plotting

fig = figure(1);
clf;

% Set line width for better visibility
fig_width = 1200;
fig_height = 500;
lineWidth = 3;
legend_font_size = 20;
label_font_size = 20;
title_font_size = 25;

colors = src.utils.get_colors();

% X method - blue
loglog(ls, conds_double_X, '-', 'DisplayName', 'X-double', 'LineWidth', lineWidth, 'Color', colors.myblue);
hold on;
loglog(ls, conds_single_X, '--s', 'DisplayName', 'X-single', 'LineWidth', lineWidth, 'Color', colors.myblue);
loglog(ls, conds_half_X, '-.o', 'DisplayName', 'X-half', 'LineWidth', lineWidth, 'Color', colors.myblue);

% CGS method - green
loglog(ls, conds_double_cgs, '-', 'DisplayName', 'CGS-double', 'LineWidth', lineWidth, 'Color', colors.mygreen);
loglog(ls, conds_single_cgs, '--s', 'DisplayName', 'CGS-single', 'LineWidth', lineWidth, 'Color', colors.mygreen);
loglog(ls, conds_half_cgs, '-.o', 'DisplayName', 'CGS-half', 'LineWidth', lineWidth, 'Color', colors.mygreen);

% MGS method - red
loglog(ls, conds_double_mgs, '-', 'DisplayName', 'MGS-double', 'LineWidth', lineWidth, 'Color', colors.myred);
loglog(ls, conds_single_mgs, '--s', 'DisplayName', 'MGS-single', 'LineWidth', lineWidth, 'Color', colors.myred);
loglog(ls, conds_half_mgs, '-.o', 'DisplayName', 'MGS-half', 'LineWidth', lineWidth, 'Color', colors.myred);

% Hessenberg method - purple
loglog(ls, conds_double_hessenberg, '-', 'DisplayName', 'Hessenberg-double', 'LineWidth', lineWidth, 'Color', colors.mypurple);
loglog(ls, conds_single_hessenberg, '--s', 'DisplayName', 'Hessenberg-single', 'LineWidth', lineWidth, 'Color', colors.mypurple);
loglog(ls, conds_half_hessenberg, '-.o', 'DisplayName', 'Hessenberg-half', 'LineWidth', lineWidth, 'Color', colors.mypurple);

title('Condition Number vs Kernel Length Scale', 'FontSize', title_font_size);
xlabel('Kernel Length Scale', 'FontSize', label_font_size);
ylabel('Condition Number', 'FontSize', label_font_size);

% Set y-axis limit to 10000
ylim([1, 10000]);

grid on;
% Increase legend font size
legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100, 100, fig_width, fig_height];

saveas(fig, 'test_10_condition_numbers.png');

% Plot number of stable columns
fig2 = figure(2);
clf;

% X method - blue
semilogx(ls, ks_double_X, '-', 'DisplayName', 'X-double', 'LineWidth', lineWidth, 'Color', colors.myblue);
hold on;
semilogx(ls, ks_single_X, '--s', 'DisplayName', 'X-single', 'LineWidth', lineWidth, 'Color', colors.myblue);
semilogx(ls, ks_half_X, '-.o', 'DisplayName', 'X-half', 'LineWidth', lineWidth, 'Color', colors.myblue);

% CGS method - green
semilogx(ls, ks_double_cgs, '-', 'DisplayName', 'CGS-double', 'LineWidth', lineWidth, 'Color', colors.mygreen);
semilogx(ls, ks_single_cgs, '--s', 'DisplayName', 'CGS-single', 'LineWidth', lineWidth, 'Color', colors.mygreen);
semilogx(ls, ks_half_cgs, '-.o', 'DisplayName', 'CGS-half', 'LineWidth', lineWidth, 'Color', colors.mygreen);

% MGS method - red
semilogx(ls, ks_double_mgs, '-', 'DisplayName', 'MGS-double', 'LineWidth', lineWidth, 'Color', colors.myred);
semilogx(ls, ks_single_mgs, '--s', 'DisplayName', 'MGS-single', 'LineWidth', lineWidth, 'Color', colors.myred);
semilogx(ls, ks_half_mgs, '-.o', 'DisplayName', 'MGS-half', 'LineWidth', lineWidth, 'Color', colors.myred);

% Hessenberg method - purple
semilogx(ls, ks_double_hessenberg, '-', 'DisplayName', 'Hessenberg-double', 'LineWidth', lineWidth, 'Color', colors.mypurple);
semilogx(ls, ks_single_hessenberg, '--s', 'DisplayName', 'Hessenberg-single', 'LineWidth', lineWidth, 'Color', colors.mypurple);
semilogx(ls, ks_half_hessenberg, '-.o', 'DisplayName', 'Hessenberg-half', 'LineWidth', lineWidth, 'Color', colors.mypurple);

title('Number of Stable Columns vs Kernel Length Scale', 'FontSize', title_font_size);
xlabel('Kernel Length Scale', 'FontSize', label_font_size);
ylabel('Number of Stable Columns', 'FontSize', label_font_size);

grid on;
% Increase legend font size
legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig2.Position = [600, 600, fig_width, fig_height];

saveas(fig2, 'test_10_stable_columns.png');

% Save data to log file
log_file = 'test10_data.log';
fid = fopen(log_file, 'w');

% Write header
fprintf(fid, 'Kernel_Length_Scale\t');
fprintf(fid, 'X_double_cond\tX_single_cond\tX_half_cond\t');
fprintf(fid, 'CGS_double_cond\tCGS_single_cond\tCGS_half_cond\t');
fprintf(fid, 'MGS_double_cond\tMGS_single_cond\tMGS_half_cond\t');
fprintf(fid, 'Hessenberg_double_cond\tHessenberg_single_cond\tHessenberg_half_cond\t');
fprintf(fid, 'X_double_k\tX_single_k\tX_half_k\t');
fprintf(fid, 'CGS_double_k\tCGS_single_k\tCGS_half_k\t');
fprintf(fid, 'MGS_double_k\tMGS_single_k\tMGS_half_k\t');
fprintf(fid, 'Hessenberg_double_k\tHessenberg_single_k\tHessenberg_half_k\n');

% Write data
for i = 1:length(ls)
   fprintf(fid, '%.6f\t', ls(i));
   fprintf(fid, '%.6f\t%.6f\t%.6f\t', conds_double_X(i), conds_single_X(i), conds_half_X(i));
   fprintf(fid, '%.6f\t%.6f\t%.6f\t', conds_double_cgs(i), conds_single_cgs(i), conds_half_cgs(i));
   fprintf(fid, '%.6f\t%.6f\t%.6f\t', conds_double_mgs(i), conds_single_mgs(i), conds_half_mgs(i));
   fprintf(fid, '%.6f\t%.6f\t%.6f\t', conds_double_hessenberg(i), conds_single_hessenberg(i), conds_half_hessenberg(i));
   fprintf(fid, '%d\t%d\t%d\t', ks_double_X(i), ks_single_X(i), ks_half_X(i));
   fprintf(fid, '%d\t%d\t%d\t', ks_double_cgs(i), ks_single_cgs(i), ks_half_cgs(i));
   fprintf(fid, '%d\t%d\t%d\t', ks_double_mgs(i), ks_single_mgs(i), ks_half_mgs(i));
   fprintf(fid, '%d\t%d\t%d\n', ks_double_hessenberg(i), ks_single_hessenberg(i), ks_half_hessenberg(i));
end

fclose(fid);
fprintf('Data saved to %s\n', log_file);

return;

function [Q, R] = dummy_orth(Q, precision_output)
   % Dummy orthogonalization, difectly get Q as output
   Q = precision_output(Q);
   R = [];
end

function [V, D] = dummy_rr(A, V)
   % Dummy Rayleigh-Ritz projection, difectly get V as output
   D = [];
end

function [cond_Q, k] = test_orth_X(K, k, maxiter, iter, seed, ...
   precision_subspace_compute, precision_subspace_output, ...
   precision_mgs_compute, precision_mgs_output)
   
   orth_function = @(A) dummy_orth(A, precision_mgs_output);
   rr_function = @dummy_rr;
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                                    'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [Q, ~] = ...
      src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                                       'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   k = size(Q, 2);
   cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_cgs(K, k, maxiter, iter, seed, ...
   precision_subspace_compute, precision_subspace_output, ...
   precision_cgs_compute, precision_cgs_output)

src.qr.cgs([], 'precision_compute', precision_cgs_compute, 'precision_output', precision_cgs_output, 'check_params', true);
orth_function = @(A) src.qr.cgs(A, 'precision_compute', precision_cgs_compute, 'precision_output', precision_cgs_output);
rr_function = @dummy_rr;
src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
[Q, ~] = ...
src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
k = size(Q, 2);
cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_cgs_custom_select_tol(K, k, maxiter, iter, seed, ...
   precision_subspace_compute, precision_subspace_output, ...
   precision_cgs_compute, precision_cgs_output, ...
   select_tol)

src.qr.cgs([], 'precision_compute', precision_cgs_compute, 'precision_output', precision_cgs_output, 'select_tol', select_tol, 'check_params', true);
orth_function = @(A) src.qr.cgs(A, 'precision_compute', precision_cgs_compute, 'precision_output', precision_cgs_output, 'select_tol', select_tol);
rr_function = @dummy_rr;
src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
[Q, ~] = ...
src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
k = size(Q, 2);
cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_mgs(K, k, maxiter, iter, seed, ...
                                                 precision_subspace_compute, precision_subspace_output, ...
                                                 precision_mgs_compute, precision_mgs_output)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output);
   rr_function = @dummy_rr;
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [Q, ~] = ...
      src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   k = size(Q, 2);
   cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_mgs_custom_select_tol(K, k, maxiter, iter, seed, ...
                                                 precision_subspace_compute, precision_subspace_output, ...
                                                 precision_mgs_compute, precision_mgs_output, ...
                                                 select_tol)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'select_tol', select_tol, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'select_tol', select_tol);
   rr_function = @dummy_rr;
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [Q, ~] = ...
      src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   k = size(Q, 2);
   cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_hessenberg(K, k, maxiter, iter, seed, ...
   precision_subspace_compute, precision_subspace_output, ...
   precision_mgs_compute, precision_mgs_output)

   src.qr.hessenberg([], 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.hessenberg(A, 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output);
   rr_function = @dummy_rr;
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
   'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [Q, ~] = ...
   src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
   'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   k = size(Q, 2);
   cond_Q = cond(double(Q));
end

function [cond_Q, k] = test_orth_hessenberg_custom_select_tol(K, k, maxiter, iter, seed, ...
   precision_subspace_compute, precision_subspace_output, ...
   precision_mgs_compute, precision_mgs_output, ...
   select_tol)

   src.qr.hessenberg([], 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'select_tol', select_tol, 'check_params', true);
   orth_function = @(A) src.qr.hessenberg(A, 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'select_tol', select_tol);
   rr_function = @dummy_rr;
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
   'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [Q, ~] = ...
   src.subspace.subspace_iter_eig(K, size(K, 1), k, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
   'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   k = size(Q, 2);
   cond_Q = cond(double(Q));
end