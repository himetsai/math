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

padded_target = zeros(row_TG + 7, col_TG + 7);
padded_target(5:4 + row_TG, 5:4 + col_TG) = target;

prior_BG = row_BG / (row_BG + row_FG);
prior_FG = row_FG / (row_BG + row_FG);

% pick cheetah if (p(x | grass) / p(x | cheetah)) < threshold
threshold = prior_FG / prior_BG;

feature_BG = zeros(64, 1);
feature_FG = zeros(64, 1);

for r = 1:1:row_BG
    maxVal = max(training_BG(r));
    secVal = 0;
    secPos = 0;
    for c = 1:1:col_BG
        if (training_BG(r, c) < maxVal && training_BG(r, c) > secVal)
            secVal = training_BG(r, c);
            secPos = c;
        end
    end
    feature_BG(secPos) = feature_BG(secPos) + 1;
end

for r = 1:1:row_FG
    maxVal = max(training_FG(r));
    secVal = 0;
    secPos = 0;
    for c = 1:1:col_FG
        if (training_FG(r, c) < maxVal && training_FG(r, c) > secVal)
            secVal = training_FG(r, c);
            secPos = c;
        end
    end
    feature_FG(secPos) = feature_FG(secPos) + 1;
end

cprob_BG = feature_BG / sum(feature_BG);
cprob_FG = feature_FG / sum(feature_FG);

A = zeros(row_TG, col_TG);

for r = 1:row_TG
    for c = 1:col_TG
        block = padded_target(r:r + 7, c:c + 7);
        dctBlock = abs(dct2(block));
        maxVal = max(max(dctBlock));
        secVal = 0;
        x = 0;
        for i = 1:8
            for j = 1:8
                if dctBlock(i, j) < maxVal && dctBlock(i, j) > secVal
                    secVal = dctBlock(i, j);
                    x = zigzag(i, j);
                end
            end
        end
        A(r, c) = int8(cprob_BG(x)/cprob_FG(x) <= threshold);
    end
end

subplot(1, 3, 1);
imagesc(target);
axis off
colormap("gray");
axis equal tight;

subplot(1, 3, 2);
imagesc(mask);
axis off
colormap(gray(255));
axis equal tight;

subplot(1, 3, 3);
imagesc(A);
axis off
colormap(gray(255));
axis equal tight;

error = 0;
for r = 1:row_TG
    for c = 1:col_TG
        if (A(r, c) ~= mask(r, c))
            error = error + 1;
        end
    end
end
error_rate = error / (row_TG * col_TG);