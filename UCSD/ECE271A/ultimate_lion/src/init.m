function [p, mu, sigma] = init(dataset, C)
    [row, col] = size(dataset);
    data_mean = sum(dataset, 1) / row;

    mu = zeros(col, C);
    mu(:, 1) = data_mean';

    sigma = zeros(col, C);
    
    count = 1;
    epoch = 1000;
    
    while count < C
        temp_mu = zeros(col, C);
        for i=1:count
            temp_mu(:, 2 * i - 1) = mu(:, i);
            temp_mu(:, 2 * i) = mu(:, i) .* 1.01;
        end
        mu = temp_mu;
        T = zeros(row, C);
        count = count * 2;
        
        for itr=1:epoch
            T = zeros(row, C);
            for i=1:row
                closest = 1;
                for j=1:count
                    if sum((dataset(i, :)' - mu(:, j)).^2) < sum((dataset(i, :)' - mu(:, closest)).^2)
                        closest = j;
                    end
                end
                T(i, closest) = 1;
            end
            mu = dataset' * T ./ sum(T);
            mu(isnan(mu)) = 0;
        end
        p = sum(T) ./ sum(sum(T));
        for i=1:row
            c = find(T(i, :));
            sigma(:, c) = sigma(:, c) + (dataset(i, :)' - mu(:, c)).^2;
        end
        sigma = sigma ./sum(T);
        sigma(isnan(sigma)) = 0;
    end
    sigma(sigma < 0.0001) = 0.0001;
end