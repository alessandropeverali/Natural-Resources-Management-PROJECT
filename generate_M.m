function [M, Y] = generate_M(x,y,ar)
%[M, Y] = generate_M(x,y,ar)
%it generate M and Y for an arx(n) shifted of 3
M=[x(1:end-3,:) y(1:end-3)];
if ar>1
    for i=2:ar
        size(M(1:end-1,:));
        size(x(i:end-3,:));
        M=[M(1:end-1,:), x(i:end-3,:), y(i:end-3)];
    end
end
Y=y(ar+3:end);
end