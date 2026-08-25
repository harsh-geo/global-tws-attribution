function test_shap()
    X = rand(100, 5);
    y = sum(X, 2) + randn(100, 1);
    rf = TreeBagger(10, X, y, 'Method', 'regression');
    predict_fcn = @(X) predict(rf, X);
    
    explainer = shapley(predict_fcn, X);
    disp('Class of ShapleyValues:');
    disp(class(explainer.ShapleyValues));
    disp('Size of ShapleyValues:');
    disp(size(explainer.ShapleyValues));
end
test_shap();
