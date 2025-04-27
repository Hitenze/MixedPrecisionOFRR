function [] = plot_pts(X, Xk, Xk2, newfig)
%% function [] = plot_pts(X, Xk, Xk2, newfig)
% Plot 2D or 3D points, highlighting additional point sets if provided
%
% Inputs:
%   X: 2D or 3D points matrix (n x d)
%   Xk: (Optional) 2D or 3D points to be highlighted in red (k x d)
%   Xk2: (Optional) 2D or 3D points to be highlighted in black (m x d)
%   newfig: (Optional) Whether to create a new figure, default is 1
%
% Example:
%   X = src.kernel.generate_pts(100, 20);
%   % Select 10 points using farthest point sampling (if available)
%   Xk = X(1:10, :);  % For illustration
%   Xk2 = X(1:5, :);  % For illustration
%   src.kernel.plot_pts(X, Xk, Xk2);

   [~, d] = size(X);

   if nargin < 2
      Xk = [];
   end

   if nargin < 3
      Xk2 = [];
   end

   if nargin < 4
      newfig = 1;
   end

   if(d == 2)
      plot2d_pts(X, Xk, Xk2, newfig);
   elseif(d == 3)
      plot3d_pts(X, Xk, Xk2, newfig);
   else
      warning("dimension not supported\n");
   end

end

function [] = plot2d_pts(X, Xk, Xk2, newfig)
   n = size(X,1);

   if nargin < 4
      newfig = 1;
   end

   smallp = 5;
   largep = 20;
   largerp = 50;

   if newfig
      figure;
   end

   plot(X(1,1),X(1,2),'b.', 'MarkerSize', smallp);
   hold on;

   for i = 2:n
      plot(X(i,1),X(i,2),'b.', 'MarkerSize', smallp);
   end

   if ~isempty(Xk)
      k = size(Xk,1);
      for i = 1:k
         plot(Xk(i,1),Xk(i,2),'r.', 'MarkerSize', largep);
      end
   end

   if ~isempty(Xk2)
      k = size(Xk2,1);
      for i = 1:k
         plot(Xk2(i,1),Xk2(i,2),'k.', 'MarkerSize', largerp);
      end
   end

   xlim1 = min(X(:,1));
   xlim2 = max(X(:,1));
   ylim1 = min(X(:,2));
   ylim2 = max(X(:,2));
   xlim([xlim1 xlim2]);
   ylim([ylim1 ylim2]);
end

function [] = plot3d_pts(X, Xk, Xk2, newfig)
   n = size(X,1);

   if nargin < 4
      newfig = 1;
   end

   smallp = 5;
   largep = 20;
   largerp = 50;

   if newfig
      figure;
   end
   
   plot3(X(1,1),X(1,2),X(1,3),'b.', 'MarkerSize', smallp);
   hold on;

   for i = 2:n
      plot3(X(i,1),X(i,2),X(i,3),'b.', 'MarkerSize', smallp);
   end

   if ~isempty(Xk)
      k = size(Xk,1);
      for i = 1:k
         plot3(Xk(i,1),Xk(i,2),Xk(i,3),'r.', 'MarkerSize', largep);
      end
   end

   if ~isempty(Xk2)
      k = size(Xk2,1);
      for i = 1:k
         plot3(Xk2(i,1),Xk2(i,2),Xk2(i,3),'k.', 'MarkerSize', largerp);
      end
   end
end

