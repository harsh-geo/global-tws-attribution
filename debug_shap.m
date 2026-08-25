load('outputs/tables/shap_results.mat');
fid = fopen('shap_debug.txt', 'w');
for i = 1:length(shap_results)
    if isempty(shap_results{i})
        fprintf(fid, 'Basin index %d is empty.\n', i);
    else
        fprintf(fid, 'Basin %d:\n', shap_results{i}.basin_id);
        fprintf(fid, '  class: %s\n', class(shap_results{i}.shap_values));
        fprintf(fid, '  size: %d %d\n', size(shap_results{i}.shap_values, 1), size(shap_results{i}.shap_values, 2));
    end
end
fclose(fid);
