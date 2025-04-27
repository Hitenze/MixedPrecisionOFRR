%% Test 00
% This test uses standard MGS to show the importance of using orthgonal basis
% FP16 + FP32 compute for real CUDA core computation

n = 1000;
l = 10;
s = 1e-01;
m = 40;
k = 20;
iter = 2;
maxiter = 3;
reorth_tol = 1/sqrt(2); 
seed = 42;
rng(seed);
X = src.kernel.generate_pts(n,20);
K = src.kernel.gaussian(X,X,l,1,s);

tic;
[eig_double_double, k_double_double] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                              @double, @double, @double, @double);
toc;

tic;
[eig_single_single, k_single_single] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                              @single, @single, @single, @single);
toc;

tic;
[eig_single_double, k_single_double] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                              @single, @single, @double, @double);
toc;

tic;
[eig_half_half, k_half_half] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                      @single, @half, @single, @half);
toc;

tic;
[eig_half_single, k_half_single] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                          @single, @half, @single, @single);
toc;

tic;
[eig_half_double, k_half_double] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                          @single, @half, @double, @double);
toc;

k_new = min([k_double_double, k_single_single, k_single_double, k_half_half, k_half_single, k_half_double]);
if k_new ~= k
   warning('insufficient k obtained');
   k = k_new;
   eig_double_double = eig_double_double(1:k);
   eig_single_single = eig_single_single(1:k);
   eig_single_double = eig_single_double(1:k);
   eig_half_half = eig_half_half(1:k);
   eig_half_single = eig_half_single(1:k);
   eig_half_double = eig_half_double(1:k);
end

tic;
eig_true = sort(real(eig(K)), 'descend');
eig_true = eig_true(1:k);
toc;

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

% Plotting with specified colors and line styles
semilogy(abs(eig_double_double - eig_true)./abs(eig_true), 'DisplayName', 'double-double', 'LineWidth', lineWidth, 'Color', colors.myblue);
hold on;
semilogy(abs(eig_single_single - eig_true)./abs(eig_true), 'DisplayName', 'single-single', 'LineWidth', lineWidth, 'Color', colors.myred, 'LineStyle', '--');
semilogy(abs(eig_single_double - eig_true)./abs(eig_true), 'DisplayName', 'single-double', 'LineWidth', lineWidth, 'Color', colors.myred);
semilogy(abs(eig_half_half - eig_true)./abs(eig_true), 'DisplayName', 'half-half', 'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '-.');
semilogy(abs(eig_half_single - eig_true)./abs(eig_true), 'DisplayName', 'half-single', 'LineWidth', lineWidth, 'Color', colors.mypurple, 'LineStyle', '--');
semilogy(abs(eig_half_double - eig_true)./abs(eig_true), 'DisplayName', 'half-double', 'LineWidth', lineWidth, 'Color', colors.mypurple);

title('Relative Error of Approximated Eigenvalues', 'FontSize', title_font_size);
xlabel('Eigenvalue index', 'FontSize', label_font_size);
ylabel('Relative error', 'FontSize', label_font_size);

% Increase legend font size
legend('show', 'FontSize', legend_font_size, 'Location', 'best');

fig.Position = [100, 100, fig_width, fig_height];

saveas(fig, 'test_00.png');

function [eig_values, k] = test_subspace_iter_eig(K, m, k, maxiter, iter, seed, ...
                                                 precision_subspace_compute, precision_subspace_output, ...
                                                 precision_mgs_compute, precision_mgs_output)

   src.qr.mgs([], 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output, 'check_params', true);
   orth_function = @(A) src.qr.mgs(A, 'precision_compute', precision_mgs_compute, 'precision_output', precision_mgs_output);
   src.rr.eig_rr([], [], 'precision_matvec_compute', precision_mgs_compute, 'precision_matvec_output', precision_mgs_compute, ...
                  'precision_A_matvec_compute', precision_subspace_compute, 'precision_A_matvec_output', precision_subspace_compute, ...
                  'precision_eig', @double, 'check_params', true);
   rr_function = @(A, V) src.rr.eig_rr(A, V, 'precision_matvec_compute', precision_subspace_compute, 'precision_matvec_output', precision_subspace_compute, ...
                                        'precision_A_matvec_compute', precision_subspace_compute, 'precision_A_matvec_output', precision_subspace_compute, ...
                                        'precision_eig', @double);
   src.subspace.subspace_iter_eig([], [], [], 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function, 'check_params', true);
   [~, D] = ...
      src.subspace.subspace_iter_eig(K, size(K, 1), m, 'maxiter', maxiter, 'step_size', iter, 'precision_compute', precision_subspace_compute, 'precision_output', precision_subspace_output, ...
                           'seed', seed, 'function_orth', orth_function, 'function_rr', rr_function);
   eig_values = sort(real(diag(D)), 'descend');
   k = min(k, length(eig_values));
   eig_values = eig_values(1:k);

end