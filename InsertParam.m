function [Mat,Dim] = InsertParam(Par,Mat,Dim)
%InsertParam
% Par is the parameter matrix you want to add at the end
% Mat is the matrix
% Dim is the matrix that traces the dimensions of all the Par used

% The minimum number of columns it produces for Mat is 2
Dim = [Dim;size(Par)];
max_col = size(Mat,2);

% Minimum number of columns is 2
if (size(Par,2) == 1)
    Par(:,2) = zeros(size(Par,1),1);
end

if (size(Par,2) > size(Mat,2))
    NumZeroCol = size(Par,2) - size(Mat,2);
    Mat(:,max_col+1:max_col+NumZeroCol) = zeros(size(Mat,1),NumZeroCol);
end
clear NumZeroCol
if (size(Mat,2) > size(Par,2))
     NumZeroCol = size(Mat,2) - size(Par,2);
     Par(:,size(Par,2)+1:size(Par,2)+NumZeroCol) = zeros(size(Par,1),NumZeroCol);
end
clear max_col, clear NumZeroCol
Mat = [Mat; Par];
end

