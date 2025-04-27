function [U, S, V] = svd_rr(A, U, V, varargin)
%% function [U, S, V] = svd_rr(A, U, V, varargin)
% Rayleigh-Ritz method for singular value decomposition
%
% Inputs:
%   A: matrix
%   U: left approximate singular vectors
%   V: right approximate singular vectors
%   varargin: options
%       'precision_matvec_compute': precision of the matrix-vector multiplication for computation
%               Optional. Default is same as U.
%       'precision_matvec_output': precision of the matrix-vector multiplication for output
%               Optional. Default is same as precision_matvec_compute.
%       'precision_A_matvec_compute': precision of the matrix-vector multiplication for A
%               Optional. Default is same as precision_matvec_compute.
%       'precision_A_matvec_output': precision of the matrix-vector multiplication for A
%               Optional. Default is same as precision_A_matvec_compute.
%       'precision_eig': precision of the eigenvalues
%               Optional. Default is double.
%       'use_generalized': bool, use generalized eigenvalue problem
%               Optional. Default is false.
%       'generalized_tol': tolerance for stability of the generalized eigenvalue problem
%               Optional. Default is 1e-10 for double, 1e-5 for single, 1e-3 for half.
%       'matvec': function for the matrix-vector product
%               Optional. In the form of @(A, trans, x, 'precision_compute', precision_compute, 'precision_output', precision_output), where
%               A is the matrix, trans is 'N' or 'T', x is the vector, and
%               precision_compute is the precision of the computation (optional), and
%               precision_output is the precision of the output vector (optional).
%               Default is the standard dense matrix-vector product.
%
% Outputs:
%   U: left singular vectors (in input precision)
%   S: diagonal matrix of singular values (in precision_eig)
%   V: right singular vectors (in input precision)
%
% Example:
%   A = rand(100, 80);              % Matrix to decompose
%   U = rand(100, 20);              % Initial left vectors
%   V = rand(80, 20);               % Initial right vectors
%   [U, S, V] = src.rr.svd_rr(A, U, V, 'precision_matvec_compute', 'double', 'precision_matvec_output', 'single');

   [precision_matvec_compute, precision_matvec_output, precision_A_matvec_compute, precision_A_matvec_output, precision_eig, use_generalized, generalized_tol, check_params, matvec] = parse_options(varargin{:});
   if check_params
      U = [];
      S = [];
      V = [];
      return;
   end

   precision_V = str2func(class(V));
   precision_U = str2func(class(U));
   assert(isequal(precision_V, precision_U), 'U and V must be of the same precision when using Rayleigh-Ritz SVD');

   % Process inputs 
   AV = precision_matvec_compute(matvec(A, 'N', precision_A_matvec_compute(V), 'precision_compute', precision_A_matvec_compute, 'precision_output', precision_A_matvec_output));
   UAV = precision_eig(src.mvops.gemm(U, 'T', AV, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output));
   
   if use_generalized
      UU = precision_eig(src.mvops.gemm(U, 'T', U, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output));
      VV = precision_eig(src.mvops.gemm(V, 'T', V, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output));
      
      ku = size(UU, 1);
      kv = size(VV, 1);
      k = min(ku, kv);
      A_extended = [ precision_eig(zeros(ku, ku)),    UAV; 
                     UAV',                            precision_eig(zeros(kv, kv))];
      M_extended = [ UU,                              precision_eig(zeros(ku, kv));
                     precision_eig(zeros(kv, ku)),    VV];
                     
      [V_extended, D_extended] = eig(A_extended, M_extended + generalized_tol * precision_eig(eye(size(A_extended))));
      [~, idx] = sort(real(diag(D_extended)), 'descend');
      D_extended = D_extended(idx, idx);
      V_extended = V_extended(:, idx);
      
      S = D_extended(1:k, 1:k);
      Z = V_extended(1:ku, 1:k);
      W = V_extended(ku+1:end, 1:k);
   else
      [Z, S, W] = svd(UAV);
   end
   
   % Convert eigenvectors back to working precision
   Z = precision_matvec_output(Z);
   W = precision_matvec_output(W);
   
   % Compute final singular vectors
   U = src.mvops.gemm(U, 'N', Z, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output);
   U = precision_V(precision_matvec_output(sqrt(2)) * U);
   
   V = src.mvops.gemm(V, 'N', W, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output);
   V = precision_V(precision_matvec_output(sqrt(2)) * V);
end

function [precision_matvec_compute, precision_matvec_output, precision_A_matvec_compute, precision_A_matvec_output, precision_eig, use_generalized, generalized_tol, check_params, matvec] = parse_options(varargin)
   if nargin > 1
      for i = 1:2:nargin
         switch varargin{i}
            case 'precision_matvec_compute'
               precision_matvec_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_matvec_output'
               precision_matvec_output = src.utils.parse_precision(varargin{i+1});
            case 'precision_A_matvec_compute'
               precision_A_matvec_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_A_matvec_output'
               precision_A_matvec_output = src.utils.parse_precision(varargin{i+1});
            case 'precision_eig'
               precision_eig = src.utils.parse_precision(varargin{i+1});
            case 'use_generalized'
               use_generalized = varargin{i+1};
            case 'generalized_tol'
               generalized_tol = varargin{i+1};
            case 'check_params'
               check_params = varargin{i+1};
            case 'matvec'
               matvec = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   if ~exist('precision_eig', 'var')
      precision_eig = @double;
   end
   if ~exist('precision_matvec_compute', 'var') && ~exist('precision_matvec', 'var')
      precision_matvec_compute = precision_eig;
   end
   if ~exist('precision_matvec_output', 'var')
      precision_matvec_output = precision_matvec_compute;
   end
   if ~exist('precision_A_matvec_compute', 'var')
      precision_A_matvec_compute = precision_matvec_compute;
   end
   if ~exist('precision_A_matvec_output', 'var')
      precision_A_matvec_output = precision_A_matvec_compute;
   end
   if ~exist('use_generalized', 'var')
      use_generalized = false;
   end
   if ~exist('generalized_tol', 'var')
      if isequal(precision_eig, @double)
         generalized_tol = 1e-10;
      elseif isequal(precision_eig, @single)
         generalized_tol = 1e-5;
      else
         generalized_tol = 1e-3;
      end
   end
   if ~exist('check_params', 'var')
      check_params = false;
   end
   if ~exist('matvec', 'var')
      matvec = @src.mvops.gemm;
   end
   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for Rayleigh-Ritz SVD:\n');
      fprintf(' -- precision_matvec_compute: %s\n', char(precision_matvec_compute));
      fprintf(' -- precision_matvec_output: %s\n', char(precision_matvec_output));
      fprintf(' -- precision_A_matvec_compute: %s\n', char(precision_A_matvec_compute));
      fprintf(' -- precision_A_matvec_output: %s\n', char(precision_A_matvec_output));
      fprintf(' -- precision_eig: %s\n', char(precision_eig));
      fprintf(' -- use_generalized: %d\n', use_generalized);
      fprintf(' -- generalized_tol: %e\n', generalized_tol);
      fprintf(' -- matvec: %s\n', char(matvec));
      fprintf('--------------------------------\n');
   end
end