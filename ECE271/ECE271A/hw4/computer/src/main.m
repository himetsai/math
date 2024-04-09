load('../dataset/TrainingSamplesDCT_subsets_8.mat');
alpha = load('../dataset/Alpha.mat');
alpha = alpha.alpha;
strat_1 = load('../dataset/Prior_1.mat');
strat_2 = load('../dataset/Prior_2.mat');
zigzag = load('../dataset/Zig-Zag Pattern.txt');
cheetah = imread('../dataset/cheetah.bmp');
cheetah_mask = imread('../dataset/cheetah_mask.bmp');
target = im2double(cheetah);
mask = im2double(cheetah_mask);

dataset_BG = {D1_BG, D2_BG, D3_BG, D4_BG};
dataset_FG = {D1_FG, D2_FG, D3_FG, D4_FG};
strats = {strat_1, strat_2};

[row_TG, col_TG] = size(target);
[~, alpha_dim] = size(alpha);

zigzag = zigzag + 1;

figure;
for d = 1:4
  training_BG = dataset_BG{d};
  training_FG = dataset_FG{d};

  [row_BG, col_BG] = size(training_BG);
  [row_FG, col_FG] = size(training_FG);

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

  mle_result = mle(training_BG, training_FG) * ones(alpha_dim);

  for s = 1:2
    strat = strats{s};
    
    bayes_result = zeros(1, alpha_dim);
    map_result = zeros(1, alpha_dim);
    
    for alp = 1:alpha_dim
    
        sigma0 = alpha(alp) * diag(strat.W0);
        
        mu_bayes_FG = sigma0 * inv(sigma0 + cov_FG/row_FG) * mean_FG' + ...
            cov_FG/row_FG * inv(sigma0 + cov_FG/row_FG) * strat.mu0_FG';
        mu_bayes_BG = sigma0 * inv(sigma0 + cov_BG/row_BG) * mean_BG' + ...
            cov_BG/row_BG * inv(sigma0 + cov_BG/row_BG) * strat.mu0_BG';
        
        sigma_bayes_FG = sigma0 * inv(sigma0 + cov_FG/row_FG) * cov_FG/row_FG;
        sigma_bayes_BG = sigma0 * inv(sigma0 + cov_BG/row_BG) * cov_BG/row_BG;
        
        A_bayes = zeros(row_TG, col_TG);
        A_map = zeros(row_TG, col_TG);
        
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
                A_bayes(r, c) = int8(mvn(X, mu_bayes_BG', cov_BG + sigma_bayes_BG)/ ...
                    mvn(X, mu_bayes_FG', cov_FG + sigma_bayes_FG) <= threshold);
                A_map(r, c) = int8(mvn(X, mu_bayes_BG', cov_BG)/ ...
                    mvn(X, mu_bayes_FG', cov_FG) <= threshold);
            end
        end
        
        error_bayes = 0;
        error_map = 0;
        for r = 1:row_TG
            for c = 1:col_TG
                if (A_bayes(r, c) ~= mask(r, c))
                    error_bayes = error_bayes + 1;
                end
        
                if (A_map(r, c) ~= mask(r, c))
                    error_map = error_map + 1;
                end
            end
        end
        
        error_rate_bayes = error_bayes / (row_TG * col_TG);
        error_rate_map = error_map / (row_TG * col_TG);
        bayes_result(alp) = error_rate_bayes;
        map_result(alp) = error_rate_map;
    end
    
    subplot(4, 2, 2 * d + s - 2);
    
    plot(alpha, bayes_result, 'b');
    hold on;
    
    plot(alpha, map_result, 'r');
    hold on;
    
    plot(alpha, mle_result, 'g');
    
    set(gca, 'XScale', 'log');
    
    title("Strategy: " + s + ", Dataset: " + d, 'Interpreter', 'latex');
    
    xlabel("\alpha");
    ylabel("Probability of Error", 'Interpreter', 'latex');
    
    legend({"Bayes-BDR", "MAP-BDR", "MLE-BDR"}, 'Location','southeast', ...
        'Interpreter', 'latex', 'FontSize', 6);
  end
end