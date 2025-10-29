function [p, mu, sigma] = EM(dataset, p0, mu0, sigma0, C)
    [row, col] = size(dataset);
    
    epoch = 100;
    p = p0;
    mu = mu0;
    sigma = sigma0;
    
    for itr=1:epoch
        h = zeros(row, C);
        for i=1:row
            for j=1:C
                h(i, j) = mvn(dataset(i, :), mu(:, j)', ...
                    diag(sigma(:, j)')) * p(j);
            end
            h(i, :) = h(i, :) ./ sum(h(i, :));
        end
    
        for j=1:C
            p(j) = sum(h(:, j)) / row;
            temp_mu = zeros(col, 1);
            for i=1:row
                temp_mu = temp_mu + h(i, j) * dataset(i, :)';
            end
            
            mu(:, j) = temp_mu / sum(h(:, j));
            temp_sigma = zeros(col, 1);
            for i=1:row
                temp_sigma = temp_sigma + h(i, j) * ((dataset(i, :)' - mu(:, j)).^2);
            end
            sigma(:, j) = temp_sigma / sum(h(:, j));
            sigma(sigma < 0.0001) = 0.0001;
        end
    end
end