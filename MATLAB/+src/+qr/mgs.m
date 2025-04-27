function varargout = mgs(A, varargin)
%% function [Q,R] = mgs(A, varargin)
%% function [Q,R,select] = mgs(A, varargin)
% Compute the QR factorization of A using modified Gram-Schmidt orthogonalization.
%
% Inputs:
%   A: matrix for the QR factorization
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as A.
%       'precision_output': precision for output
%               Optional. Default is same as precision_compute.
%       'tol': tolerance for the QR factorization
%               Optional. Default is machine epsilon.
%       'reorth_tol': tolerance for reorthogonalization
%               Optional. Default is 1/sqrt(2).
%       'select_tol': tolerance for selecting columns
%               Optional. Default is machine epsilon.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   [Q,R]: A = Q * R
%      Q: orthogonal matrix
%      R: upper triangular matrix
%   [Q,R,select]: A = Q * R
%      Q: Full size linearly independent matrix
%      R: Full size upper triangular matrix
%      select: boolean vector. False indicates near zero columns
%
% Example:
%   [Q,R] = src.qr.mgs(A, 'precision_compute', 'single', 'precision_output', 'half', 'tol', 1e-5);
%   [Q,R,select] = src.qr.mgs(A, 'reorth_tol', -1); % No reorthogonalization

   [precision_compute, precision_output, orth_tol, reorth_tol, select_tol, check_params] = parse_options(varargin{:});
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

   A = precision_output(A);
   Q = precision_output(zeros(m,n));
   R = precision_output(zeros(n,n));

   select = true(n,1);
   
   for i = 1:n
      v = A(:,i);
      
      % mark this for re-orth
      normv = src.mvops.norm2(v, 'precision_compute', precision_compute, 'precision_output', precision_output);

      if normv < select_tol
         select(i) = false;
         continue;
      end

      % MGS
      for j = 1:i-1
         if select(j)
            R(j,i) = src.mvops.dot(v, Q(:,j), 'precision_compute', precision_compute, 'precision_output', precision_output);
            v = precision_output(precision_compute(v) - precision_compute(R(j,i))*precision_compute(Q(:,j)));
         end
      end

      % re-orth
      t = src.mvops.norm2(v, 'precision_compute', precision_compute, 'precision_output', precision_output);
      while(reorth_tol >= 0 && (t >= orth_tol && t < reorth_tol*normv))
         normv = t;
         for j = 1:i-1
            if select(j)
               r = src.mvops.dot(v, Q(:,j), 'precision_compute', precision_compute, 'precision_output', precision_output);
               v = precision_output(precision_compute(v) - precision_compute(r)*precision_compute(Q(:,j)));
               R(j,i) = precision_output(precision_compute(R(j,i)) + precision_compute(r));
            end
         end
         t = src.mvops.norm2(v, 'precision_compute', precision_compute, 'precision_output', precision_output);
      end

      if (t < select_tol)
         select(i) = false;
      else
         Q(:,i) = precision_output(precision_compute(v)/precision_compute(t));
         R(i,i) = t;
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

function [precision_compute, precision_output, orth_tol, reorth_tol, select_tol, check_params] = parse_options(varargin)
   
   if nargin > 1
      for i = 1:2:length(varargin)
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'tol'
               orth_tol = varargin{i+1};
            case 'reorth_tol'
               reorth_tol = varargin{i+1};
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
   if ~exist('orth_tol', 'var')
      orth_tol = src.utils.eps(char(precision_output));
   else
      orth_tol = precision_output(orth_tol);
   end
   if ~exist('reorth_tol', 'var')
      reorth_tol = 1.0/sqrt(2.0);
   else
      reorth_tol = precision_output(reorth_tol);
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
      fprintf('Parameters for MGS:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- orth_tol: %e\n', orth_tol);
      fprintf(' -- reorth_tol: %e\n', reorth_tol);
      fprintf(' -- select_tol: %e\n', select_tol);
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end