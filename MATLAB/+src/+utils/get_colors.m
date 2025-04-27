function colors = get_colors()
%% function colors = get_colors()
% get custom colors for plotting
%
% Output
%   colors: struct with custom colors
%       Available colors:
%           myblue, myred, myyellow, mypurple, mygreen, mygray
%
% Example
%   colors = src.utils.get_colors();
%   plot(1:10, rand(1,10), 'Color', colors.myblue);

   myblue   = [102, 196, 255] / 255;
   myred    = [239, 77, 59] / 255;
   myyellow = [244, 205, 113] / 255;
   mypurple = [175, 88, 186] / 255;
   mygreen  = [154, 224, 133] / 255;
   mygray   = [220, 220, 220] / 255;

   colors = struct('myblue', myblue, 'myred', myred, 'myyellow', myyellow, 'mypurple', mypurple, 'mygreen', mygreen, 'mygray', mygray);

end