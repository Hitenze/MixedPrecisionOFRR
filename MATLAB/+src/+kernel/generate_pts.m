function [X] = generate_pts(n, type)
%% function X = generate_pts(n, type)
% Generate dataset with n points, with "average density" of 1
% This means that the average points in unit length/area/volume is roughly 1
%
% Inputs:
%   n: Number of points
%   type: Type of dataset in two digits code [xy]
%      x: Dimension of the dataset
%         2: 2D dataset
%         3: 3D dataset
%      y: Type of the dataset
%         0: Rectangle / cube
%         1: Disk / ball
%         2: Circle / sphere
%         3: Regular grid
%         6: 2D only circle with a cluster inside
%         7: 2D smiley face
%         8: 3D test nonuniform
%
% Outputs:
%   X: n × d matrix, n is the number of points, d is the dimension
%
% Example:
%   % Generate 100 points in 2D rectangle
%   X = src.kernel.generate_pts(100, 20);
%   % Generate 1000 points on 3D sphere
%   X = src.kernel.generate_pts(1000, 32);

   %%-----------------------------------------------------------------
   %  2D rectangle, area of a rectangle with edge length r is r^2
   %  we have n points inside this area, and thus we hope
   %  to have r^2 = n, and thus r = n^(1/2)
   %%-----------------------------------------------------------------
   if type == 20
      X = n^(1/2)*(rand(n, 2));
      return;
   end

   %%-----------------------------------------------------------------
   %  3D cube, volumn of a cube with edge length r is r^3
   %  we have n points inside this volumn, and thus we hope
   %  to have r^3 = n, and thus r = n^(1/3)
   %%-----------------------------------------------------------------
   if type == 30
      X = n^(1/3)*(rand(n, 3));
      return;
   end

   %%-----------------------------------------------------------------
   %  2D disk, area of a disk with radious r is pi*r^2
   %  we have n points inside this area, and thus we hope
   %  to have pi*r^2 = n, and thus r = (n/pi)^(1/2)
   %%-----------------------------------------------------------------
   if type == 21
      theta = 2*pi*rand(n,1);
      radii = (n/pi)^(1/2)*(rand(n,1).^(1/2));
      
      x = radii.*sin(theta);
      y = radii.*cos(theta);
      
      X = [x,y];
      return;
   end

   %%-----------------------------------------------------------------
   %  3D ball, volumn of a ball with radious r is 4/3*pi*r^3
   %  we have n points inside this volumn, and thus we hope
   %  to have 4/3*pi*r^3 = n, and thus r = (3n/4/pi)^(1/3)
   %%-----------------------------------------------------------------
   if type == 31
      rvals = 2*rand(n,1)-1;
      elevation = asin(rvals);
      
      azimuth = 2*pi*rand(n,1);
      
      radii = (3*n/4/pi)^(1/3)*(rand(n,1).^(1/3));
      
      [x,y,z] = sph2cart(azimuth,elevation,radii);
      X = [x,y,z];
      return;
   end

   %%-----------------------------------------------------------------
   %  2D circle, the circumference of a circle with radious r is 2*pi*r
   %  we have n points on this circle, and thus we hope
   %  to have 2*pi*r = n, and thus r = n/2/pi
   %%-----------------------------------------------------------------
   if type == 22
      theta = 2*pi*rand(n,1);
      radii = n/2/pi;
      
      x = radii.*sin(theta);
      y = radii.*cos(theta);
         
      X = [x,y];
      return;
   end

   %%-----------------------------------------------------------------
   %  3D sphere, the surface area of a sphere with radious r is 4*pi*r^2
   %  we have n points on this sphere, and thus we hope
   %  to have 4*pi*r^2 = n, and thus r = (n/4/pi)^(1/2)
   %%-----------------------------------------------------------------
   if type == 32
      rvals = 2*rand(n,1)-1;
      elevation = asin(rvals);
      
      azimuth = 2*pi*rand(n,1);
      
      radii = (n/4/pi)^(1/2);
      
      [x,y,z] = sph2cart(azimuth,elevation,radii);
      X = [x,y,z];
      return;
   end

   %%-----------------------------------------------------------------
   %  2D regular grid, the area of a grid with grid size r is r^2
   %  we have n points inside this area, and thus we hope
   %  to have r^2 = n, and thus r = n^(1/2)
   %  however, we have exactly n^(1/2) points in each direction
   %%-----------------------------------------------------------------
   if type == 23
      nx = round(sqrt(n));
      if(nx^2 ~= n)
         warning("sqrt(n) must be an integer");
         X = [];
      else
         X = zeros(n,2);
         rn = 1;
         for i = 1:nx
            for j = 1:nx
               X(rn,1) = j;
               X(rn,2) = i;
               rn = rn + 1;
            end
         end
      end
      return;
   end

   %%-----------------------------------------------------------------
   %  3D regular grid, the volumn of a grid with grid size r is r^3
   %  we have n points inside this volumn, and thus we hope
   %  to have r^3 = n, and thus r = n^(1/3)
   %  however, we have exactly n^(1/3) points in each direction
   %%-----------------------------------------------------------------
   if type == 33
      nx = round(n^(1/3));
      if(nx^3 ~= n)
         warning("n^(1/3) must be an integer");
         X = [];
      else
         X = zeros(n,3);
         rn = 1;
         for i = 1:nx
            for j = 1:nx
               for k = 1:nx
                  X(rn,1) = j;
                  X(rn,2) = i;
                  X(rn,3) = k;
                  rn = rn + 1;
               end
            end
         end
      end
      return;
   end

   %%-----------------------------------------------------------------
   %   2D circle with a cluster of points at the center
   %%-----------------------------------------------------------------
   if type == 26
      % a cluster of points with many outlayers
      n1 = ceil(n/3);
      n2 = n - n1;
      scalen = 0.1;

      theta = 2*pi*rand(n1,1);
      radii = n/pi/2;
      
      x = radii.*sin(theta);
      y = radii.*cos(theta);
      
      
      theta1 = 2*pi*rand(n2,1);
      radii1 = (n1/pi/scalen)^(1/2)*(rand(n2,1).^(3));
      
      x1 = radii1.*sin(theta1);
      y1 = radii1.*cos(theta1);

      X = [x(1:end-5),y(1:end-5);x1(1:end-5),y1(1:end-5);x(end-4:end),y(end-4:end);x1(end-4:end),y1(end-4:end)];
      return;
   end

   %%-----------------------------------------------------------------
   %   2D smiley face
   %%-----------------------------------------------------------------
   if type == 27
      % a circle as the face
      % two clusters of eyes
      % a cluster of mouth
      n_face = ceil(n/4);
      n_eye = ceil(n/4);
      n_mouth = n - n_face - 2*n_eye;

      % first create the face
      theta_face = 2*pi*rand(n_face,1);
      radii_face = n/pi/2*(1+1e-01*rand(n_face,1));
      
      x_face = radii_face.*sin(theta_face);
      y_face = radii_face.*cos(theta_face);
      
      X_face = [x_face,y_face];
      
      theta_eye1 = 2*pi*rand(n_eye,1);
      theta_eye2 = 2*pi*rand(n_eye,1);

      radii_eye1 = (n_face/pi/5)^(1/2)*(rand(n_eye,1).^(1/2)).*(1+n/pi/2*rand(n_eye,1)*1e-02);
      radii_eye2 = (n_face/pi/5)^(1/2)*(rand(n_eye,1).^(1/2)).*(1+n/pi/2*rand(n_eye,1)*1e-02);
      
      x_eye1 = radii_eye1.*sin(theta_eye1);
      y_eye1 = radii_eye1.*cos(theta_eye1);

      x_eye2 = radii_eye2.*sin(theta_eye2);
      y_eye2 = radii_eye2.*cos(theta_eye2);

      X_eye1 = [x_eye1,y_eye1] + [0.5,0.5]*n/pi/2;
      X_eye2 = [x_eye2,y_eye2] + [-0.5,0.5]*n/pi/2;

      % finally create the mouse
      theta_mouth = pi+pi/2*(rand(n_mouth,1)-0.5);
      radii_mouth = n/pi/4*(1+1e-02*rand(n_mouth,1));

      x_mouth = radii_mouth.*sin(theta_mouth);
      y_mouth = radii_mouth.*cos(theta_mouth);

      X_mouth = [x_mouth,y_mouth] + [0,-0.2]*n/pi/2;

      X = [X_face;X_eye1;X_eye2;X_mouth]*0.5;
      return;
   end

   %%-----------------------------------------------------------------
   %  2D nonuniform disk, area of a disk with radious r is pi*r^2
   %  we have n points inside this area, and thus we hope
   %  to have pi*r^2 = n, and thus r = (n/pi)^(1/2)
   %%-----------------------------------------------------------------
   if type == 28
      theta = 2*pi*rand(n,1);
      radii = (n/pi)^(1/2)*(rand(n,1).^(3));
      
      x = radii.*sin(theta);
      y = radii.*cos(theta);
      
      X = [x,y];
      return;
   end

   %%-----------------------------------------------------------------
   %  3D test
   %%-----------------------------------------------------------------
   if type == 38
      rvals = 2*rand(n,1)-1;
      elevation = asin(rvals);
      
      azimuth = 2*pi*rand(n,1);
      
      radii = (3*n/4/pi)^(1/3)*(min(abs(randn(n,1)),1).^1);
      
      [x,y,z] = sph2cart(azimuth,elevation,radii);
      X = [x,y,z]*diag([2.0,1.0,0.5]);
      return;
   end

   %%-----------------------------------------------------------------
   %  3D test
   %%-----------------------------------------------------------------
   if type == 39
      n1 = n;
      k = floor(n/8);
      n = n1 - k;
      rvals = 2*rand(n,1)-1;
      elevation = asin(rvals);
      
      azimuth = 2*pi*rand(n,1);
      
      radii = (3*n/4/pi)^(1/3)*(rand(n,1).^(1/3))/100;
      
      [x,y,z] = sph2cart(azimuth,elevation,radii);
      
      n = k;
      rvals = 2*rand(n,1)-1;
      elevation = asin(rvals);
      
      azimuth = 2*pi*rand(n,1);
      
      radii = (n/4/pi)^(1/2);
      
      [x1,y1,z1] = sph2cart(azimuth,elevation,radii);

      X = [x,y,z;x1,y1,z1];

      return;
   end

   warning("type unsupported");
   X = [];
end