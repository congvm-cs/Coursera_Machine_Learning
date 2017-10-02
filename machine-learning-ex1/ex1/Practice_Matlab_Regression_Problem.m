%% Reset config
clear all; close all; clc;
%% Loading data
dataset = load('ex1data1.txt', ',');
X = dataset(:, 1);
y = dataset(:, 2);
sprintf('size of X: %d\n', size(X))
sprintf('size of y: %d\n', size(y))
