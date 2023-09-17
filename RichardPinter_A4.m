%{
    Richard Pinter - Assignment 4
    Simple decision stump with Titanic data.

    Objective:
    The aim of this assignment is to implement a simple decision stump
    to predict survival on the Titanic. 
    We will be employing the CART (Classification and Regression Trees)
    algorithm to construct this basic decision tree. 
    The stump will utilise features such as Sex, Age,
    and Pclass to make survival predictions.

    Data Features:
    - Sex: Gender of the passenger (1 for male, 0 for female).
    - Age: Age of the passenger.
    - Pclass: Ticket class (1 for first class, 2 for second class, 3 for third class).

    Methodology:
    1. Load the Titanic dataset.
    2. Preprocess the data.
    3. Build the decision stump.
    4. Make predictions on new data points.

    Prediction Example:
    The decision stump will be used to predict the
    survival outcome of a specific passenger:
    A 22-year-old male with a Pclass value of 3.
    The feature set would be encoded as [1, 22, 3].

    Note: This is a simplified model, and the accuracy may be limited due
    to the use of only a single feature for splitting.
%}


% Avoid superimposed operations and close previous figures
clc; clear; close all 


%%% Load the data in and pre-processing
% Load data
data = readtable('titanic.csv');

% Preprocess data
data.Sex = double(strcmp(data.Sex, 'male'));
data.Age(isnan(data.Age)) = median(data.Age, 'omitnan');

% Extract features and labels
X = data(:, {'Sex', 'Age', 'Pclass'});
X = table2array(X);
y = data.Survived;


%%% Build decision stump and predict the outcome

% Build the decision stump
stump = build_stump(X, y);

% Data point: 22-year-old male with Pclass value of 3
% Encoding: [Sex, Age, Pclass] where Sex: 1 for male, 0 for female
single_data_point = [1, 22, 3];

% Make a single prediction using the decision stump
single_prediction = predict_stump(stump, single_data_point);

% Display the prediction
if single_prediction == 1
    fprintf('The individual is predicted to survive.\n');
else
    fprintf('The individual is predicted to not survive.\n');
end

%%% Functions 

% Creates stump
function [stump] = build_stump(X, y)

    % Initialise variables
    [n, m] = size(X);
    best_gini = Inf;
    best_feature = 0;
    best_value = 0;
    
    % Loop through each of the features
    for feature = 1:m
        values = unique(X(:, feature));
        % Loop through each of the values inside a given feature
        for value = values'
            % Split the data to left and right node
            left_split = y(X(:, feature) < value);
            right_split = y(X(:, feature) >= value);
            
            % Calculate gini impurity for this value for this feature
            gini = gini_impurity(left_split, right_split);
            
            % Update best feature so far
            if gini < best_gini
                best_gini = gini;
                best_feature = feature;
                best_value = value;
            end
        end
    end
    
    % Return stump as a struct
    stump = struct(...
        'feature', best_feature,...
        'value', best_value,...
        'gini', best_gini,...
        'left_label', majority_label(y(X(:, best_feature) < best_value)),...
        'right_label', majority_label(y(X(:, best_feature) >= best_value)));
end

% Helper calculates gini_impurity
function [gini] = gini_impurity(left, right)

    % Probability of left and probability of right
    total = length(left) + length(right);
    p_left = length(left) / total;
    p_right = 1 - p_left;
    
    % Gini imputiy score for the left and right node
    gini_left = 1 - sum((histcounts(left, 'Normalization', 'probability').^2));
    gini_right = 1 - sum((histcounts(right, 'Normalization', 'probability').^2));
    
    % Returns gini impurity
    gini = p_left * gini_left + p_right * gini_right;
end

% Helper calculates majority of the label
function [label] = majority_label(split)
    if isempty(split)
        label = 0;
    else
        label = mode(split);
    end
end

% Helper predicts target for new sata
function [label] = predict_stump(stump, x)
    if x(stump.feature) < stump.value
        label = stump.left_label;
    else
        label = stump.right_label;
    end
end