load('../dataset/TrainingSamplesDCT_8.mat');
zigzag = load('../dataset/Zig-Zag Pattern.txt');
cheetah = imread('../dataset/cheetah.bmp');
cheetah_mask = imread('../dataset/cheetah_mask.bmp');
target = im2double(cheetah);
mask = im2double(cheetah_mask);


% Configuration
training_BG = TrainsampleDCT_BG;
training_FG = TrainsampleDCT_FG;

E = zeros(8, 64);
best8 = [1 25 27 32 33 45 46 48];

for i = 1:8
    E(i, best8(i)) = 1;
end

training_BG = training_BG * E';
training_FG = training_FG * E';

[row_BG, col_BG] = size(training_BG);
[row_FG, col_FG] = size(training_FG);
[row_TG, col_TG] = size(target);

zigzag = zigzag + 1;

prior_BG = row_BG / (row_BG + row_FG);
prior_FG = row_FG / (row_BG + row_FG);

% pick cheetah if (p(x | grass) / p(x | cheetah)) < threshold
threshold = prior_FG / prior_BG;

C = 8;


% Initialization
[pi_BG, mu_BG, sigma_BG] = init(training_BG, C);
[pi_FG, mu_FG, sigma_FG] = init(training_FG, C);
disp("Parameters initialized");


% Run EM
[pi_BG, mu_BG, sigma_BG] = EM(training_BG, pi_BG, mu_BG, sigma_BG, C);

disp("Finished EM for BG");

[pi_FG, mu_FG, sigma_FG] = EM(training_FG, pi_FG, mu_FG, sigma_FG, C);

disp("Finished EM for FG");


% Convert params to mixure distribution
sq_sigma_BG = zeros(col_BG, col_BG, C);
for i = 1:C
    sq_sigma_BG(:, :, i) = diag(sigma_BG(:, i));
end

sq_sigma_FG = zeros(col_FG, col_FG, C);
for i = 1:C
    sq_sigma_FG(:, :, i) = diag(sigma_FG(:, i));
end

gm_BG = gmdistribution(mu_BG(1:col_BG, :)', sq_sigma_BG, pi_BG);
gm_FG = gmdistribution(mu_FG(1:col_FG, :)', sq_sigma_FG, pi_FG);


% Inference
A = zeros(row_TG, col_TG);

for r = 5:row_TG-3
    for c = 5:col_TG-3
        block = target(r - 4:r + 3, c - 4:c + 3);
        dctBlock = dct2(block);
        X = zeros(1, 64);
        for i = 1:8
            for j = 1:8
                X(zigzag(i, j)) = dctBlock(i, j);
            end
        end
        X = X * E';
        A(r, c) = int8(pdf(gm_BG, X) / pdf(gm_FG, X) <= threshold);
    end
end


% Calculate PoE
error = 0;
for r = 1:row_TG
    for c = 1:col_TG
        if (A(r, c) ~= mask(r, c))
            error = error + 1;
        end
    end
end

error_rate = error / (row_TG * col_TG);

disp("Finished inference. Error rate is " + error_rate * 100 + "%");


% Plot
figure;

subplot(1, 3, 1);
imagesc(target);
axis off
colormap(gray(255));
axis equal tight;
title("Original", 'Interpreter', 'latex', 'FontSize', 20);

subplot(1, 3, 2);
imagesc(mask);
axis off
colormap(gray(255));
axis equal tight;
title("Truth", 'Interpreter', 'latex', 'FontSize', 20);

subplot(1, 3, 3);
imagesc(A);
axis off
colormap(gray(255));
axis equal tight;
title("Prediction (error rate: " + round(error_rate * 100, 2) + "\%)", ...
    'Interpreter', 'latex', 'FontSize', 20);