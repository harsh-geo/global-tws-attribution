load('outputs/tables/shap_results.mat');
for i = 1:length(shap_results)
    if isempty(shap_results{i})
        fprintf('Basin index %d is empty.\n', i);
    else
        fprintf('Basin %d:\n', shap_results{i}.basin_id);
        fprintf('  class(shap_values): %s\n', class(shap_results{i}.shap_values));
        fprintf('  size(shap_values): %d %d\n', size(shap_results{i}.shap_values, 1), size(shap_results{i}.shap_values, 2));
    end
end
