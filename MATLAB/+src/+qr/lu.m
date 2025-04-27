function varargout = lu(A, varargin)
%% function [L,U] = lu(A, varargin)
%% function [L,U,perm,select] = lu(A, varargin)
% Column-pivoted LU factorization for non-square matrices
%
% Inputs:
%   A: input matrix (m x n)
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as A.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%       'select_tol': tolerance for selecting columns
%               Optional. Default is machine epsilon.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   [L,U]: A = L * U
%      L: economic size pivoted lower triangular factor
%      U: economic size pivoted upper triangular factor
%   [L,U,perm,select]: P * A = L * U
%      L: full size lower triangular factor
%      U: full size upper triangular factor
%      perm: permutation vector (m x 1)
%      select: logical vector (n x 1), False indicates near zero columns
%
% Example:
%   [L,U] = src.qr.lu(A, 'precision_compute', 'single', 'precision_output', 'half');
%   [L,U,perm,select] = src.qr.lu(A, 'select_tol', 1e-6); % Use custom selection tolerance

   [precision_compute, precision_output, select_tol, check_params] = parse_options(varargin{:});
   if check_params
      switch nargout
         case 2
            varargout{1} = [];
            varargout{2} = [];
         case 4
            varargout{1} = [];
            varargout{2} = [];
            varargout{3} = [];
            varargout{4} = [];
      end
      return;
   end
   [m,n] = size(A);
   assert(m >= n, 'A must be a tall-thin matrix');
   
   % Initialize matrices in output precision
   A = precision_output(A);
   L = precision_output(zeros(m,n));
   U = precision_output(zeros(n,n));
   perm = 1:m;
   select = true(n,1);

   % Working copy of A
   A_work = A;
   
   % Main loop for LU factorization
   for j = 1:n
      % Find the largest element in current column for pivoting
      [max_val, pivot_idx] = max(abs(A_work(j:m,j)));
      pivot_idx = pivot_idx + j - 1;
      
      % Handle effectively zero columns
      if max_val < select_tol
         select(j) = false;
         U(j,j:n) = A_work(j,j:n);
         continue;
      end
      
      % Swap rows if necessary
      if pivot_idx ~= j
         A_work([j,pivot_idx],:) = A_work([pivot_idx,j],:);
         if j > 1
            L([j,pivot_idx],1:j-1) = L([pivot_idx,j],1:j-1);
         end
         perm([j,pivot_idx]) = perm([pivot_idx,j]);
      end
      
      % Store the current row in U
      U(j,j:n) = A_work(j,j:n);
      
      % Compute multipliers in compute precision and store in output precision
      L(j:m,j) = precision_output(precision_compute(A_work(j:m,j)) / precision_compute(A_work(j,j)));
      
      % Update remaining submatrix with precision control
      for i = j+1:m
         A_work(i,j+1:n) = precision_output(precision_compute(A_work(i,j+1:n)) - ...
                                          precision_compute(L(i,j)) * precision_compute(A_work(j,j+1:n)));
      end
      %A_work(j+1:m,j+1:n) = precision(A_work(j+1:m,j+1:n) - ...
      %   L(j+1:m,j) * A_work(j,j+1:n));
   end

   if nargout == 2
      rperm = zeros(m,1);
      rperm(perm) = 1:m;
      L = L(rperm,:);
      varargout{1} = L(:,select);
      varargout{2} = U(select,:);
   elseif nargout == 4
      varargout{1} = L;
      varargout{2} = U;
      varargout{3} = perm;
      varargout{4} = select;
   else
      error('Invalid number of output arguments');
   end
end

function [precision_compute, precision_output, select_tol, check_params] = parse_options(varargin)
   if nargin > 1
      for i = 1:2:length(varargin)
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'select_tol'
               select_tol = varargin{i+1};
            case 'check_params'
               check_params = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   if ~exist('precision_compute', 'var') && ~exist('precision', 'var')
      precision_compute = @double;
   end
   if ~exist('precision_output', 'var')
      precision_output = precision_compute;
   end
   if ~exist('select_tol', 'var')
      select_tol = src.utils.eps(char(precision_output));
   else
      select_tol = precision_output(select_tol);
   end
   if ~exist('check_params', 'var')
      check_params = false;
   end
   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for LU:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- select_tol: %e\n', select_tol);
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end
