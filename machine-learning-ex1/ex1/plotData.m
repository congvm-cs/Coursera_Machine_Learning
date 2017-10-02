function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
plot(x, y, 'rx');
% initialize J vals to a matrix of 0's
% J_vals = zeros(length(theta0_vals), length(theta1_vals));
% Fill out J vals
% for i = 1:length(theta0_vals)
%     for j = 1:length(theta1_vals)
%       t = [theta0_vals(i); theta1_vals(j)];
%       J_vals(i,j) = computeCost(x, y, t);
%     end
% end

% ============================================================

end
