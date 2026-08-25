fprintf('Loading shap_results...\n');
load('outputs/tables/shap_results.mat');
for i=1:length(shap_results)
    if isempty(shap_results{i}), continue; end
    fprintf('Basin %d:\n', shap_results{i}.basin_id);
    fprintf('  class: %s\n', class(shap_results{i}.shap_values));
    sz = size(shap_results{i}.shap_values);
    fprintf('  size: %d %d\n', sz(1), sz(2));
    if istable(shap_results{i}.shap_values)
        disp(shap_results{i}.shap_values(1,:));
    end
end
fprintf('Done.\n');
