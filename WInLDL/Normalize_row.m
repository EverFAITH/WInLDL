function x = Normalize_row(x)
% v = sum(x,2);
% x = x ./ v;
% x(isnan(x)) = 0;
[n,~] = size(x);
for i = 1 : n
    if norm(x(i,:)) == 0
        x(i,:) = 0;
    else
    x(i,:) = x(i,:) / norm(x(i,:));    % Normalize by row
    end
end
end
