function [V, D] = eig_rr(A, V, varargin)
%% function [V, D] = eig_rr(A, V, varargin)
% Rayleigh-Ritz method for eigenvalue problem
%
% Inputs:
%   A: matrix
%   V: approximate eigenvectors
%   varargin: options
%       'precision_matvec_compute': precision of the matrix-vector multiplication for computation
%               Optional. Default is same as V.
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
%   V: eigenvectors (in input precision)
%   D: diagonal matrix of eigenvalues (in precision_eig)
%
% Example:
%   A = rand(100, 100); A = A + A';  % Symmetric matrix
%   V = rand(100, 10);               % Initial guess
%   [V, D] = src.rr.eig_rr(A, V, 'precision_matvec', 'double', 'use_generalized', true);

   [precision_matvec_compute, precision_matvec_output, precision_A_matvec_compute, precision_A_matvec_output, precision_eig, use_generalized, generalized_tol, check_params, matvec] = parse_options(varargin{:});
   if check_params
      V = [];
      D = [];
      return;
   end

   precision_V = str2func(class(V));

   AV = precision_matvec_compute(matvec(A, 'N', precision_A_matvec_compute(V), 'precision_compute', precision_A_matvec_compute, 'precision_output', precision_A_matvec_output));
   VAV = precision_eig(src.mvops.gemm(V, 'T', AV, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output));
   if use_generalized
      VV = precision_eig(src.mvops.gemm(V, 'T', V, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output));
      [U, D] = eig(VAV, VV + generalized_tol * precision_eig(eye(size(V, 2))));
   else
      [U, D] = eig(VAV);
   end
   V = src.mvops.gemm(V, 'N', U, 'precision_compute', precision_matvec_compute, 'precision_output', precision_matvec_output);
   V = precision_V(V);
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
   if ~exist('precision_matvec_compute', 'var')
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
      fprintf('Parameters for Rayleigh-Ritz:\n');
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