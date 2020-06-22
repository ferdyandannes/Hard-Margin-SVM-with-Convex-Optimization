%% CVX SVM Library Example
clear all; close all; clc;
folder = '.\DataFolder\\';

%% Create a Basic Problem #1
T = readtable('admission.csv');
sigma = 1;
X = T(1:80,1:2);
X = table2array(X);
X_test = T(81:100,1:2);
X_test = table2array(X_test);

% Change label 0 to -1
Z = T(1:100,end);
Z = table2array(Z);
for i = 1:100
    if Z(i,end) == 0
        Z(i,end) = -1;
    end
end
Y = Z(1:80,end);
Y_test = Z(81:100, end);


%% Create a Basic Problem #2
% T = readtable('master.csv');
% sigma = 1;
% X = T(1:500,[6 11]);
% X = table2array(X);
% X_test = T(501:700,[6 11]);
% X_test = table2array(X_test);
% 
% % Change label 0 to -1
% Z = T(1:700,7);
% Z = table2array(Z);
% for i = 1:700
%     if Z(i) >= 4
%         Z(i) = -1;
%     else
%         Z(i) = 1;
%     end
% end
% Y = Z(1:500);
% Y_test = Z(501:700);



%% Find Results with the Built-in SVM as Baseline
D = fitcsvm(X,Y);
accuracy_baseline = mean(predict(D,X_test)==Y_test);

%% PRINT VARIABLE
K=X*X'; % Dot-product kernel
N = size(X,1);
disp(K)
disp(X)
disp(X')
disp(N)
disp(Y)

%% Percobaan
% Training
coba_model = coba(X,Y);
baseFileName = sprintf('coba.jpeg');
fullFileName = fullfile(folder, baseFileName);
plot_the_SVs(X, Y, coba_model, "Coba Support Vectors", fullFileName, true);
% Training Plot
baseFileName = sprintf('coba2.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_hard = svmPredictPlot(X, Y, coba_model, "Coba Training - Classification Boundary and Training Points", fullFileName, true, false);
% Testing and Test Plot with model and Data Points
baseFileName = sprintf('coba3.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_hard = svmPredictPlot(X_test, Y_test, coba_model, "Coba Testing - Classification Boundary and Testing Points", fullFileName, true, false);

%% HARD MARGIN
% Training
hard_model = hardmargin_svm(X,Y);
baseFileName = sprintf('HardMarginSVPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
plot_the_SVs(X, Y, hard_model, "Hard Margin Support Vectors", fullFileName, true);
% Training Plot
baseFileName = sprintf('HardMarginTrainingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_hard = svmPredictPlot(X, Y, hard_model, "Hard Margin Training - Classification Boundary and Training Points", fullFileName, true, false);
% Testing and Test Plot with model and Data Points
baseFileName = sprintf('HardMarginTestingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_hard = svmPredictPlot(X_test, Y_test, hard_model, "Hard Margin Testing - Classification Boundary and Testing Points", fullFileName, true, false);

%% SOFT MARGIN
soft_model = softmargin_svm(X,Y,1,1)
baseFileName = sprintf('SoftMarginSVPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
plot_the_SVs(X, Y, soft_model, "Soft Margin Support Vectors", fullFileName, true);
baseFileName = sprintf('SoftMarginTrainingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_soft = svmPredictPlot(X, Y, soft_model, "Soft Margin Training - Classification Boundary and Training Points", fullFileName, true, false);
% Testing and Test Plot with model and Data Points
baseFileName = sprintf('SoftMarginTestingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_soft = svmPredictPlot(X_test, Y_test, soft_model, "Soft Margin Testing - Classification Boundary and Testing Points", fullFileName, true, false);

%% NON LINEAR CLASSIFICATION WITH KERNEL TRICK
nonlinear_model = nonlinear_svm(X,Y,1,1);
baseFileName = sprintf('NonlinearSVPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
plot_the_SVs(X, Y, nonlinear_model, "Nonlinear Support Vectors", fullFileName, true);
baseFileName = sprintf('NonlinearTrainingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_nonlinear = svmPredictPlot(X, Y, nonlinear_model, "Nonlinear Training - Classification Boundary and Training Points", fullFileName, true, false);
% Testing and Test Plot with model and Data Points
baseFileName = sprintf('NonlinearTestingPredictionRandomPlot.jpeg');
fullFileName = fullfile(folder, baseFileName);
accuracy_nonlinear = svmPredictPlot(X_test, Y_test, nonlinear_model, "Nonlinear Testing - Classification Boundary and Testing Points", fullFileName, true, false);