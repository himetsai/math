function result = mvn(x, mean, cov)
    [~, dim] = size(mean);
    d = (x - mean) * pinv(cov) * (x - mean)';
    c = 1/sqrt((2 * pi)^dim * det(cov));
    result = c * exp(-0.5 * d);
end