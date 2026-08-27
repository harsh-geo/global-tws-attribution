

script_dir = fileparts(mfilename('fullpath'));
project_root = script_dir;

fprintf('=================================================================\n');
fprintf('  GLOBAL TWS ATTRIBUTION - RUNNING ONLY MODELING & VALIDATION    \n');
fprintf('=================================================================\n\n');

%% STEP 4: Twin RF Machine Learning Attribution
fprintf('--> Running STEP 4: Twin RF Attribution Modeling...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'modeling', 'step04_run_attribution.m'));
fprintf('✓ STEP 4 Completed in %.2f seconds.\n\n', toc(t_step));

%% STEP 4c/d: PINN and LSTM Attribution
fprintf('--> Running STEP 4c: PINN Attribution Modeling...\n');
run(fullfile(project_root, 'src', 'modeling', 'step04e_run_attribution_pinn.m'));
fprintf('--> Running STEP 4d: LSTM Attribution Modeling...\n');
run(fullfile(project_root, 'src', 'modeling', 'step04d_run_attribution_lstm.m'));

%% STEP 5: Validation, Trends, and Output Generation
fprintf('--> Running STEP 5: Validation & Trend Analysis...\n');
t_step = tic;
run(fullfile(project_root, 'src', 'validation', 'step05_validate_and_trends.m'));
fprintf('✓ STEP 5 Completed in %.2f seconds.\n\n', toc(t_step));

fprintf('=================================================================\n');
fprintf('  PIPELINE COMPLETE - Results Saved to outputs/                  \n');
fprintf('=================================================================\n');
