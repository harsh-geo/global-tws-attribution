% Test the numerical instability of Central Difference Integration
n = 100;
t = (1:n)';
true_tws = sin(2*pi*t/12); % True TWS is a sine wave
twsc = zeros(n, 1);
for i=2:n-1
    twsc(i) = (true_tws(i+1) - true_tws(i-1)) / 2;
end

% Add a tiny bit of noise to TWSC (like a model prediction would have)
twsc_pred = twsc + randn(n, 1) * 0.05;

% Reconstruct using the matrix A (as in step 04)
A = zeros(n, n);
for i = 2:n-1
    A(i, i-1) = -0.5;
    A(i, i+1) =  0.5;
end
A(1, 1) = 1; A(1, 2) = 0;
A(end, end-1) = -1; A(end, end) = 1;

B = twsc_pred;
B(1) = true_tws(1);

tws_reconstructed = A \ B;

% Now reconstruct with Tikhonov regularization to suppress zig-zag
D = eye(n);
for i=2:n
    D(i, i-1) = -1;
end
lambda = 0.1; % Regularization strength
tws_reconstructed_reg = (A'*A + lambda*(D'*D)) \ (A'*B);

figure;
plot(t, true_tws, 'k-', 'LineWidth', 2); hold on;
plot(t, tws_reconstructed, 'r--', 'LineWidth', 1.5);
plot(t, tws_reconstructed_reg, 'b-', 'LineWidth', 1.5);
legend('True TWS', 'Unstable A\B', 'Regularized');
title('Integration Instability');
saveas(gcf, 'scratch/test_integration.png');
