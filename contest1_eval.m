% THIS IS THE SCRIPT FOR CONTEST

%% Clean up
clear
close all
clc

%% Add paths and load data
load -ascii Parametri.txt

NumPar = Parametri(end,1);
Size = Parametri(end-NumPar:end-1,1:2);

ind = 1;
C = Parametri(ind:ind+Size(1,1)-1,1:Size(1,2));
ind = ind+Size(1,1);
beta_red = Parametri(ind:ind+Size(2,1)-1,1:Size(2,2));
ind = ind+Size(2,1);
beta_blue = Parametri(ind:ind+Size(3,1)-1,1:Size(3,2));
ind = ind+Size(3,1);
Best_red_arx = Parametri(ind:ind+Size(4,1)-1,1:Size(4,2));
ind = ind+Size(4,1);
Best_blue_arx = Parametri(ind:ind+Size(5,1)-1,1:Size(5,2));
ind = ind+Size(5,1);
sel_input_red = Parametri(ind:ind+Size(6,1)-1,1:Size(6,2))';
ind = ind+Size(6,1);
sel_input_blue = Parametri(ind:ind+Size(7,1)-1,1:Size(7,2))';
ind = ind+Size(7,1);
clear ind, clear Size

% Please select here the test dataset
test_set = readtable('');   % we select training_set to perform Test
col = 2:27;
test_set = table2array(test_set(:,col));
p = test_set(:,1:25);   % Input
n = test_set(:,end);  % cum flow 3 days ahead

% Mean of first 5 elements of the output
n_mean = zeros(3,1);
n_mean(1) = mean(n(1:5));
n_mean(2) = n_mean(1);
n_mean(3) = n_mean(1);

n = n(1:end-3,:);
n = [n_mean;n];

% This part is used to split the dataset into TrainValidation and Test, the results changes. Probably it is due to a dataset which is not well balanced between training and validation 
% year=length(p)/365;
% port_TrainValidation=9;
% year_TrainValidation = min(round(year*port_TrainValidation), year-1);
% p = p(year_TrainValidation*365+1:end, :);
% n = n(year_TrainValidation*365+1:end,:);

%% Nan Analysis
dim_p = size(p);
nan_idx=isnan(p);
num_p = dim_p(2);

% Substitute Nan value with the mean of the column or the previoud value
m_p = mean(p,1,'omitnan');
for i = 1:dim_p(1) % rows
    for h = 1:dim_p(2) % columns
        if nan_idx(i,h) == 1
            p(i,h) = p(i-1,h); %m_p(h);
        end
    end
end

clear nan_idx, clear nan_row, clear m_p, clear i, clear h

% NaN output (generally there isn't)
nan_idx=isnan(n);
nan_flag = 0;

m_n = mean(n,1,'omitnan');
for i = 1:length(n) % rows
        if nan_idx(i) == 1
            n(i) = n(i-1);
            nan_flag = 1; % There is at least a Nan value
        end
end

if nan_flag == 1
    disp('There is at least a Nan value in the output')
end

clear nan_idx, clear nan_flag,clear m_n, clear i, clear dim_p, clear num_p

%% Compute moving average and std deviation
T  = 365; % period (days)
window_size = 21; % window size - 1 -> 21 days
N = length(n);
t = [1 : N]';

dim_p = size(p);
num_p = dim_p(2);

% Moving average
[mi, m]=moving_average(n, T, (window_size-1)/2);
[vi, v]=moving_average((n-m).^2, T, (window_size-1)/2);

m_p=p;
v_p=p;
for i=1:num_p
    [~, m_p(:,i)]= moving_average(p(:,i), T, (window_size-1)/2);
    [~, v_p(:,i)]= moving_average((p(:,i)-m_p(:,i)).^2, T, (window_size-1)/2);
end

sigma=sqrt(v);
sigma_p=sqrt(v_p);

u=(p-m_p)./sigma_p;
x=(n-m)./sigma;

clear window_size, clear i

% clear Nan for deseasonalized variables
nan_idx=isnan(u);
dim_u = size(u);

% Change Nan with its mean or its previous value
m_u = mean(u,1,'omitnan');
for i = 1:dim_u(1) % rows
    for h = 1:dim_u(2) % columns
        if nan_idx(i,h) == 1
            u(i,h) = u(i-1,h); % m_u(h);
        end
    end
end

clear nan_idx, clear i, clear h, clear m_u

% clear Nan for output
nan_idx=isnan(x);
dim_x = length(x);
nan_flag = 0;

% Change Nan with its mean or its previous value
m_x = mean(u,1,'omitnan');
for i = 1:dim_x % rows
    if nan_idx(i) == 1
        x(i) = x(i-1); % m_x;
        nan_flag = 1;
    end
end

if nan_flag == 1
    disp('There is at least a Nan value in the output')
end

clear nan_idx, clear nan_flag, clear i, clear m_x

%% Forecasting of the 3-days inflow to Como Lake
% In this section, you have to load the model you choosed and perform the
% forecasting of the output variable on the test set.
% The predictions have to be stored in a column vector and saved as a .txt
% file named "output_forecast_GroupNumber.txt"

% K-means clustering - separation

% Computing d1 and d2
k_sel_test = [x];

d1 = [k_sel_test]-C(1,:);
d2 = [k_sel_test]-C(2,:);
d1_square=0;
d2_square=0;
h = 0;

disp('Number of Outputs+Inputs used for K-means clustering:')
for i = 1:1:size(C,2)
    d1_square = d1_square+d1(:,i).^2;
    d2_square = d2_square+d2(:,i).^2;
    h = h+1;
end
disp(h)

d1 = sqrt(d1_square);
d2 = sqrt(d2_square);
clear i, clear h

jr=1;
jb=1;

% Creating red and blue clusters (according to the training)
for i = 1:1:length(x)
    if d1(i)<d2(i)      % dry -> red cluster
        r = 1;
        for h = sel_input_red
            Cr(jr,r) = u(i,h); % Input matrix
            r = r+1;
        end
        clear h, clear r
        Dr(jr,1) = x(i); % Output matrix
        indicer(jr,1)= i;
        jr=jr+1;
        idx(i)=1;
    else
        l = 1;
        for h = sel_input_blue
            Cb(jb,l) = u(i,h); % Input matrix
            l = l+1;
        end
        clear h, clear l
        Db(jb,1) = x(i); % Output matrix
        indiceb(jb,1)=i;
        jb=jb+1;
        idx(i)=2;
    end
end

clear i,clear jr,clear jb

% Plot of the timeseries highlighting the clusters
figure
hold on
for i=1:1:length(x)
    if idx(i)==1
        plot(t(i),x(i),'r.')
    end

    if idx(i)==2
        plot(t(i),x(i),'b.')
    end
end
hold off
clear i

% Estimating the output for cluster RED
k = 3;  % We want to estimate three days after
i = Best_red_arx;
q = i;

[M_red,Yin,~] = ARX(q,k,Dr,Cr);
Y_hat_red = [Yin;M_red*beta_red];
n_hat_red = (Y_hat_red.*sigma(indicer))+m(indicer);
R2_red = 1-sum((n(indicer(length(Yin)+1:end))-n_hat_red(length(Yin)+1:end)).^2)/sum((n(indicer(length(Yin)+1:end))-m(indicer(length(Yin)+1:end))).^2);
Eout_red = sum((n(indicer(length(Yin)+1:end))-n_hat_red(length(Yin)+1:end)).^2)/length(n(indicer(length(Yin)+1:end)));

clear i, clear q, clear Yin

% Estimating the output for cluster BLUE
i = Best_blue_arx;
q = i;

[M_blue,Yin,~] = ARX(q,k,Db,Cb);
Y_hat_blue = [Yin;M_blue*beta_blue];
n_hat_blue = (Y_hat_blue.*sigma(indiceb))+m(indiceb);
R2_blue = 1-sum((n(indiceb(length(Yin)+1:end))-n_hat_blue(length(Yin)+1:end)).^2)/sum((n(indiceb(length(Yin)+1:end))-m(indiceb(length(Yin)+1:end))).^2);
Eout_blue = sum((n(indiceb(length(Yin)+1:end))-n_hat_blue(length(Yin)+1:end)).^2)/length(n(indiceb(length(Yin)+1:end)));

clear i, clear q

% RED + BLUE Reconstruction
jr = 1;
jb = 1;
for i = 1:1:length(idx)
    if idx(i)==1
        n_hat(i,1) = n_hat_red(jr);
        jr = jr+1;
        a(i) = 1; % to check if all values are 1
    end
    if idx(i)==2
        n_hat(i,1) = n_hat_blue(jb);
        jb = jb+1;
        a(i) = 1;
    end
end
clear i, clear jr, clear jb

for i = 1:1:length(a) 
    if not(a(i)==1)
        disp('A problem in reconstructing has occured!')
    end
end
clear i, clear a

% Analysis of prediction
R2 = 1-sum((n(1:end)-n_hat(1:end)).^2)/sum((n(1:end)-m(1:end)).^2);
disp('R2'),disp(R2)

figure
hold on
plot(t,n,'b')
plot(t,n_hat,'r')
legend('n','n hat')
title('Test set - prediction')
hold off

%% Save the output
save output_forecast_14.txt n_hat -ascii