function error_rate = mle(training_BG, training_FG)
    zigzag = load('../dataset/Zig-Zag Pattern.txt');
    cheetah = imread('../dataset/cheetah.bmp');
    cheetah_mask = imread('../dataset/cheetah_mask.bmp');
    target = im2double(cheetah);
    mask = im2double(cheetah_mask);
    
    zigzag = zigzag + 1;

    [row_BG, ~] = size(training_BG);
    [row_FG, ~] = size(training_FG);
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
            A(r, c) = int8(mvn(X, mean_BG, cov_BG)/ ...
                mvn(X, mean_FG, cov_FG) <= threshold);
        end
    end
    
    error_rate = 0;
    for r = 1:row_TG
        for c = 1:col_TG
            if (A(r, c) ~= mask(r, c))
                error_rate = error_rate + 1;
            end
        end
    end
    error_rate = error_rate / (row_TG * col_TG);
end


