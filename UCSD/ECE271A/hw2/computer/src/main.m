load('../dataset/TrainingSamplesDCT_8.mat');
zigzag = load('../dataset/Zig-Zag Pattern.txt');
cheetah = imread('../dataset/cheetah.bmp');
cheetah_mask = imread('../dataset/cheetah_mask.bmp');
target = im2double(cheetah);
mask = im2double(cheetah_mask);

training_BG = TrainsampleDCT_BG;
training_FG = TrainsampleDCT_FG;

zigzag = zigzag + 1;

[row_BG, col_BG] = size(training_BG);
[row_FG, col_FG] = size(training_FG);
[row_TG, col_TG] = size(target);

prior_BG = row_BG / (row_BG + row_FG);
prior_FG = row_FG / (row_BG + row_FG);

% pick cheetah if (p(x | grass) / p(x | cheetah)) < threshold
threshold = prior_FG / prior_BG;

mean_FG = zeros(1, 64);
mean_BG = zeros(1, 64);

cov_FG = zeros(64, 64);
cov_BG = zeros(64, 64);

for r = 1:row_FG
    cov_FG = cov_FG + training_FG(r,:)' * training_FG(r,:);
    mean_FG = mean_FG + training_FG(r,:);
end

for r = 1:row_BG
    cov_BG = cov_BG + training_BG(r,:)' * training_BG(r,:);
    mean_BG = mean_BG + training_BG(r,:);
end

mean_FG = mean_FG / row_FG;
mean_BG = mean_BG / row_BG;

cov_FG = (cov_FG / row_FG) - mean_FG' * mean_FG;
cov_BG = (cov_BG / row_BG) - mean_BG' * mean_BG;


figure;

for k = 1:64
    subplot(8, 8, k);
    x = linspace(min(mean_FG(k) - 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) - 3 * sqrt(cov_BG(k, k))), ...
        max(mean_FG(k) + 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) + 3 * sqrt(cov_BG(k, k))), 1000);
    p_BG = (1/(sqrt(2*pi * cov_BG(k, k)))) * exp(-0.5 * ...
        ((x - mean_BG(1,k)).^2/cov_BG(k, k)));
    p_FG = (1/(sqrt(2*pi * cov_FG(k, k)))) * exp(-0.5 * ...
        ((x - mean_FG(1,k)).^2/cov_FG(k, k)));
    plot(x, p_BG, 'LineWidth', 2);
    hold on;
    plot(x, p_FG, 'LineWidth', 2);
    title("X_{" + k + "}");
end

best8 = [1 25 27 32 33 45 46 48];
worst8 = [2 3 4 59 60 62 63 64];

figure;
for i = 1:8
    subplot(2, 4, i);
    k = best8(i);
    x = linspace(min(mean_FG(k) - 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) - 3 * sqrt(cov_BG(k, k))), ...
        max(mean_FG(k) + 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) + 3 * sqrt(cov_BG(k, k))), 1000);
    p_BG = (1/(sqrt(2*pi * cov_BG(k, k)))) * exp(-0.5 * ...
        ((x - mean_BG(1,k)).^2/cov_BG(k, k)));
    p_FG = (1/(sqrt(2*pi * cov_FG(k, k)))) * exp(-0.5 * ...
        ((x - mean_FG(1,k)).^2/cov_FG(k, k)));
    plot(x, p_BG, 'LineWidth', 2);
    hold on;
    plot(x, p_FG, 'LineWidth', 2);
    title("X_{" + k + "}");
end

figure;
for i = 1:8
    subplot(2, 4, i);
    k = worst8(i);
    x = linspace(min(mean_FG(k) - 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) - 3 * sqrt(cov_BG(k, k))), ...
        max(mean_FG(k) + 3 * sqrt(cov_FG(k, k)), ...
        mean_BG(k) + 3 * sqrt(cov_BG(k, k))), 1000);
    p_BG = (1/(sqrt(2*pi * cov_BG(k, k)))) * exp(-0.5 * ...
        ((x - mean_BG(1,k)).^2/cov_BG(k, k)));
    p_FG = (1/(sqrt(2*pi * cov_FG(k, k)))) * exp(-0.5 * ...
        ((x - mean_FG(1,k)).^2/cov_FG(k, k)));
    plot(x, p_BG, 'LineWidth', 2);
    hold on;
    plot(x, p_FG, 'LineWidth', 2);
    title("X_{" + k + "}");
end

E = zeros(8, 64);

for i = 1:8
    E(i, best8(i)) = 1;
end

mean8_FG = zeros(1, 8);
mean8_BG = zeros(1, 8);

cov8_FG = zeros(8, 8);
cov8_BG = zeros(8, 8);

for r = 1:row_FG
    v = E * training_FG(r,:)';
    cov8_FG = cov8_FG + v * v';
    mean8_FG = mean8_FG + v';
end

for r = 1:row_BG
    v = E * training_BG(r,:)';
    cov8_BG = cov8_BG + v * v';
    mean8_BG = mean8_BG + v';
end

mean8_FG = mean8_FG / row_FG;
mean8_BG = mean8_BG / row_BG;

cov8_FG = (cov8_FG / row_FG) - mean8_FG' * mean8_FG;
cov8_BG = (cov8_BG / row_BG) - mean8_BG' * mean8_BG;

A_64 = zeros(row_TG, col_TG);
A_8 = zeros(row_TG, col_TG);

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
        A_64(r, c) = int8(mvn(X, mean_BG, cov_BG)/ ...
            mvn(X, mean_FG, cov_FG) <= threshold);
        A_8(r, c) = int8(mvn(X * E', mean8_BG, cov8_BG)/ ...
            mvn(X * E', mean8_FG, cov8_FG) <= threshold);
    end
end

figure;

subplot(1, 3, 1);
imagesc(target);
axis off
colormap(gray(255));
axis equal tight;

subplot(1, 3, 2);
imagesc(mask);
axis off
colormap(gray(255));
axis equal tight;

subplot(1, 3, 3);
imagesc(A_8);
axis off
colormap(gray(255));
axis equal tight;

error64 = 0;
error8 = 0;
for r = 1:row_TG
    for c = 1:col_TG
        if (A_64(r, c) ~= mask(r, c))
            error64 = error64 + 1;
        end

        if (A_8(r, c) ~= mask(r, c))
            error8 = error8 + 1;
        end
    end
end

error_rate64 = error64 / (row_TG * col_TG);
error_rate8 = error8 / (row_TG * col_TG);
disp(error_rate64);
disp(error_rate8);