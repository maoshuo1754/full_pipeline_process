function isEqual = areMareicesEqual(A, B, tolerate)
    % judge if matrix A and B is equal;
    if size(A) ~= size(B)
        isEqual = false;
        return;
    end

    diff = abs(A - B);
    if all(diff(:) < tolerate)
        isEqual = true;
    else
        isEqual = false;
    end
end