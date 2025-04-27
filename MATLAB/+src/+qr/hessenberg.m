function varargout = hessenberg(A, varargin)
%% function [Q,R] = hessenberg(A, varargin)
%% function [Q,R,select] = hessenberg(A, varargin)
% QR factorization using Hessenberg decomposition
%
% Inputs:
%   A: matrix for the QR factorization
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
%   [Q,R]: A = Q * R
%      Q: economic size linearly independent matrix
%      R: economic size upper triangular matrix
%   [Q,R,select]: A = Q * R
%      Q: Full size linearly independent matrix
%      R: Full size upper triangular matrix
%      select: boolean vector. False indicates near zero columns
%
% Example:
%   [Q,R] = src.qr.hessenberg(A, 'precision_compute', 'single', 'precision_output', 'half');
%   [Q,R,select] = src.qr.hessenberg(A, 'select_tol', 1e-6); % Custom column selection tolerance

   [precision_compute, precision_output, select_tol, check_params] = parse_options(varargin{:});
   if check_params
      switch nargout
         case 2
            varargout{1} = [];
            varargout{2} = [];
         case 3
            varargout{1} = [];
            varargout{2} = [];
            varargout{3} = [];
      end
      return;
   end

   [m,n] = size(A);
   assert(m >= n, 'A must be a tall-thin matrix');

   % Initialize matrices in output precision
   A = precision_output(A);
   Q = precision_output(zeros(m,n));
   R = precision_output(zeros(n,n));
   perm = zeros(n,1);
   select = true(n,1);
   
   for i = 1:n
      v = A(:,i);
      for j = 1:i-1
         if select(j)
            R(j,i) = v(perm(j));
            v = precision_output(precision_compute(v) - precision_compute(R(j,i)) * precision_compute(Q(:,j)));
         end
      end

      [~, k] = max(abs(v));
      R(i,i) = v(k);
      
      if abs(R(i,i)) < select_tol
         select(i) = false;
      else
         v = precision_output(precision_compute(v) / precision_compute(R(i,i)));
         perm(i) = k;
         Q(:,i) = v;
      end
   end

   if nargout == 2
      varargout{1} = Q(:,select);
      varargout{2} = R(select,:);
   elseif nargout == 3
      varargout{1} = Q;
      varargout{2} = R;
      varargout{3} = select;
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
   if ~exist('precision_compute', 'var')
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
      fprintf('Parameters for Hessenberg:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- select_tol: %e\n', select_tol);
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end