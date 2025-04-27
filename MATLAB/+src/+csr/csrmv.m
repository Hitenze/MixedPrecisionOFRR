function [y] = csrmv(Acsr, trans, x, varargin)
%% function y = csrmv(Acsr, trans, x, varargin)
% Perform matrix-vector multiplication with a CSR matrix.
% Supports different precisions for computation and output.
% Note that the we should have:
%   compute_precision >= output_precision
%
% Inputs:
%   Acsr: CSR matrix struct
%   trans: 'N' for normal, 'T' for transpose
%   x: input vector
%   varargin: options
%       'precision_compute': precision for computation
%               Optional. Default is same as x.
%       'precision_output': precision for output
%               Optional. Default is same as x.
%       'check_params': only parse parameters and display
%               Optional. Default is false.
%
% Outputs:
%   y: output vector
%
% Example:
%   y = src.csr.csrmv(Acsr, 'N', x, 'precision_compute', 'double', 'precision_output', 'double');
%   y = src.csr.csrmv(Acsr, 'T', x, 'precision_compute', @single, 'precision_output', @half);

   if(strcmp(trans, 'N'))
      assert(size(x, 1) == Acsr.ncols, 'x must have the same number of columns as Acsr');
   elseif(strcmp(trans, 'T'))
      assert(size(x, 1) == Acsr.nrows, 'x must have the same number of rows as Acsr');
   else
      error('Invalid trans argument');
   end

   [compute_precision, output_precision] = parse_options(x, varargin{:});

   x = compute_precision(x);
   nvecs = size(x, 2);

   if(strcmp(char(compute_precision), 'double'))
      aa = Acsr.a_double;
   elseif(strcmp(char(compute_precision), 'single'))
      aa = Acsr.a_single;
   elseif(strcmp(char(compute_precision), 'half'))
      aa = Acsr.a_half;
   end

   if strcmp(trans, 'N')
      y = compute_precision(zeros(Acsr.nrows, nvecs));
      for i = 1:Acsr.nrows
         colidx = Acsr.j(Acsr.i(i):Acsr.i(i+1)-1);
         if(strcmp(char(compute_precision), 'half'))
            for j = 1:nvecs
               y(i,j) = sum(aa(Acsr.i(i):Acsr.i(i+1)-1) .* x(colidx,j));
            end
         else
            y(i,:) = aa(Acsr.i(i):Acsr.i(i+1)-1)' * x(colidx,:);
         end
      end
   elseif strcmp(trans, 'T')
      y = compute_precision(zeros(Acsr.ncols, nvecs));
      for i = 1:Acsr.nrows
         colidx = Acsr.j(Acsr.i(i):Acsr.i(i+1)-1);
         if(strcmp(char(compute_precision), 'half'))
            for j = 1:nvecs
               y(colidx,j) = y(colidx,j) + sum(aa(Acsr.i(i):Acsr.i(i+1)-1)' .* x(i,j));
            end
         else
            y(colidx,:) = y(colidx,:) + aa(Acsr.i(i):Acsr.i(i+1)-1) * x(i,:);
         end
      end
   else
      error('Invalid trans argument');
   end

   y = output_precision(y);

end


function [precision_compute, precision_output] = parse_options(x, varargin)
   if nargin > 2
      for i = 1:2:nargin-1
         switch varargin{i}
            case 'precision_compute'
               precision_compute = src.utils.parse_precision(varargin{i+1});
            case 'precision_output'
               precision_output = src.utils.parse_precision(varargin{i+1});
            case 'check_params'
               check_params = varargin{i+1};
            otherwise
               error('Invalid option: %s', varargin{i});
         end
      end
   end
   input_precision = src.utils.parse_precision(class(x));
   if ~exist('precision_compute', 'var')
      precision_compute = input_precision;
   end
   if ~exist('precision_output', 'var')
      precision_output = input_precision;
   end

   if(strcmp(char(precision_output), 'double'))
      assert(strcmp(char(precision_compute), 'double'), 'compute_precision must be double if output_precision is double');
   elseif(strcmp(char(precision_output), 'single'))
      assert(strcmp(char(precision_compute), 'single') || strcmp(char(precision_compute), 'double'), ... 
             'compute_precision must be single or double if output_precision is single');
   end

   if ~exist('check_params', 'var')
      check_params = false;
   end
   if check_params
      fprintf('--------------------------------\n');
      fprintf('Parameters for csrmv:\n');
      fprintf(' -- precision_compute: %s\n', char(precision_compute));
      fprintf(' -- precision_output: %s\n', char(precision_output));
      fprintf(' -- check_params: %d\n', check_params);
      fprintf('--------------------------------\n');
   end
end