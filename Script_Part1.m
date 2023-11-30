%% Project part 1. 
% THIS IS THE MAIN SCRIPT (yes for REPORT, no for CONTEST)

clear 
close all
clc



%% Import the dataset to matlab
% This simulation takes several time to be performed (around 15 minutes).
% For this reason we decided to disable the non-linear model part (it is not our
% prefered choice). But if you want to simulate it, you can change NNswitch from zero to 1.
% Now it takes less than a minute.
NNswitch=0; 

training_set = readtable('training_set.csv');
% column | variable
% 1  = date / day
% 2  = lake Como inflow
% 3  = release Premadio
% 4  = release Isolato Spluga
% 5  = release S. Bernardo
% 6  = release Ganda
% 7  = release Vedello + Armisa
% 8  = release Lanzada
% 9  = release Gerola
% 10 = temperature Aprica
% 11 = precipitation Aprica
% 12 = temperature Gerola
% 13 = precipitation Gerola
% 14 = temperature Oga
% 15 = precipitation Oga
% 16 = temperature Palu
% 17 = precipitation Palu
% 18 = temperature Spluga
% 19 = precipitation Spluga
% 20 = temperature Vercana
% 21 = precipiation Vercana
% 22 = temperature Sondrio
% 23 = agregated precipitation in upstream basin
% 24 = SWE Adda_lac
% 25 = SWE Mera
% 26 = SWE Adda_prelac
% 27 = cum flow 3 days ahead

col = 2:27;
train_set = table2array(training_set(:,col));
p_all=train_set(:,1:25);
n_all=train_set(:,end);

% Mean of first 5 elements of the output
nall_mean = zeros(3,1);
nall_mean(1) = mean(n_all(1:5));
nall_mean(2) = nall_mean(1);
nall_mean(3) = nall_mean(1);

n_all = n_all(1:end-3,:);
n_all = [nall_mean;n_all];

table(date(1:10), p_all(1:10,1), n_all(1:10),'VariableNames', {'Date' 'Inflow (input)' 'Cum_Inflow (OUTPUT)'})

% % This part is used to split the dataset into TrainValidation and Test, the results changes. Probably it is due to a dataset which is not well balanced between training and validation 
% year=length(p_all)/365;
% port_TrainValidation=9;
% year_TrainValidation = min(round(year*port_TrainValidation), year-1);
% p_all = p_all(1:year_TrainValidation*365, :);
% n_all = n_all(1:year_TrainValidation*365,:);

clear year*
clear port*

%% NAN Analysis
dim_ptot = size(p_all);
nan_idx=isnan(p_all);
nan_col=sum(nan_idx);
nan_row=sum(nan_idx, 2);

figure
subplot(2,1,1)
plot (nan_col, '.','MarkerSize',18);
title('Number of nan in the columns')
xlabel('columns')
ylabel('Number of nan')
subplot(2,1,2)
plot (nan_row, '.','MarkerSize',18);
xlabel('rows')
ylabel('Number of nan')
title('Number of nan in the rows')


tot_nan=sum(nan_row);

T = 365;
% Substitute Nan value with the previous value of the column
% m_ptot = mean(p_all,1,'omitnan');
for i = 1:dim_ptot(1) % rows
    for h = 1:dim_ptot(2) % colomns
        if nan_idx(i,h) == 1
            p_all(i,h) = p_all(i-1,h); %m_ptot(h);
        end
    end
end

clear nan_idx, clear nan_col, clear nan_row, clear m_ptot, clear i, clear h

% No NaN in the output
nan_idx=isnan(n_all);
nan_col=sum(nan_idx);
nan_row=sum(nan_idx, 2);
tot_nan=sum(nan_row);

clear nan_idx, clear nan_col, clear nan_row



%% PCA
% Used to reduce the input dataset, but after an analysis we decided to avoid PCA.

% [IU,ES,VI]=svd(p_all);
% for i=1:15
%     gamma(i) = ES(i,i);
% end
% h=1:1:15;
% figure
% plot(h,gamma,'.')
% 
% % according to the plot I choose k=2 (as principal component)
% Vk = VI(:,1:2);
% Z = p_all*Vk;
% figure
% plot(Z(:,1)*Vk(:,1)',Z(:,2)*Vk(:,2)','.')
% 
% p_all = Z;

%% train_test separation

year=length(p_all)/365;
port_train=0.7;
year_train=min(round(year*port_train), year-1);
p = p_all(1:year_train*365, :);
n = n_all(1:year_train*365,:);
p_test=p_all(year_train*365+1:end,:);
n_test=n_all(year_train*365+1:end,:);
clear trai*_set
clear port_train
clear col
clear year*

%% PLOT TIME-SERIES


num_p=size(p);
num_p=num_p(2);
num_ptest = size(p_test);
num_ptest = num_ptest(2);

Ntrain = length(n);
t = [1 : Ntrain]'; % indices t = 1 , 2 , ... , 365 , 366 , 367 , ...

figure
plot(t, n)
xlabel('time [days]')
ylabel('inflow [m^3/s]')
title('timeseries')

% statistics
n_min   = min(n);
n_max   = max(n);
n_range = n_max - n_min;
n_mean  = mean(n);
n_var   = var(n);

%  -------------------------------------
%%   CYCLOSTATIONARY MEAN AND VARIANCE
%  -------------------------------------

% periodicity 
T  = 365; % period (days)
tt = repmat([1:365]' , Ntrain/T, 1 ) ;
% indices tt = 1 , 2 , ... , 365 , 1 , 2 , ...

figure
plot(tt, n, '.')
xlabel('time (1 year) [days]')
ylabel('inflow [m^3/s]')
title('timeseries - window of 1 year')

% reshape the vector n containing the inflow data
Q = reshape(n, T, Ntrain/T);
% cyclo-stationary mean
Cm = mean(Q, 2); % Cm = mean(Q')';
% cyclo-stationary variance
Cv = var(Q, 0, 2); % Cv = var(Q')';

% graphical analysis
figure
hold on
plot(tt, n, '.')
plot(Cm, 'r', 'LineWidth', 2)
legend('observed flow', 'cyclo mean')
xlabel('time (1 year) [days]')
ylabel('flow [m^3/s]')
title('timeseries - window of 1 year')
hold off

%  -------------------------------
%%   MOVING AVERAGE AND VARIANCE
%  -------------------------------

% compute and plot periodic mean 


window_size=21;
[mi, m]=moving_average(n, T, (window_size-1)/2);
[vi, v]=moving_average((n-m).^2, T, (window_size-1)/2);
[mi_test, m_test]=moving_average(n_test, T, (window_size-1)/2);
[vi_test, v_test]=moving_average((n_test-m_test).^2, T, (window_size-1)/2);

m_p=p;
v_p=p;
m_ptest=p_test;
v_ptest=p_test;
for i=1:num_p
    [~, m_p(:,i)]=moving_average(p(:,i), T, (window_size-1)/2);
    [~, v_p(:,i)]=moving_average((p(:,i)-m_p(:,i)).^2, T, (window_size-1)/2);
    [~, m_ptest(:,i)]=moving_average(p_test(:,i), T, (window_size-1)/2);
    [~, v_ptest(:,i)]=moving_average((p_test(:,i)-m_ptest(:,i)).^2, T, (window_size-1)/2);
end

sigma=sqrt(v);
sigma_p=sqrt(v_p);
sigma_test=sqrt(v_test);
sigma_ptest=sqrt(v_ptest);
u=(p-m_p)./sigma_p;
x=(n-m)./sigma;
u_test=(p_test-m_ptest)./sigma_ptest;
x_test=(n_test-m_test)./sigma_test;

% reshape the vector n containing the inflow data
Qx = reshape(x, T, Ntrain/T);
% cyclo-stationary mean
Cmx = mean(Qx, 2); % Cm = mean(Q')';

% Comparison of moving average and mean
figure
subplot(2,1,1)
hold on
plot(tt, n, '.')
plot(mi, 'r', 'LineWidth', 2)
title('time series with moving average')
xlabel('time (1 year) [days]')
ylabel('inflow [m^3/s]')
hold off
subplot(2,1,2)
hold on
plot(tt,x,'.')
plot(Cmx,'r','LineWidth',2)
title('time series deseasonalized')
xlabel('time (1 year) [days]')
ylabel('inflow [m^3/s]')
hold off

% Plot of the variance of the deseasonalized variable
[mxi,mx]= moving_average(x,T,(window_size-1)/2);
[vxi,vx]= moving_average((x-mx).^2,T,(window_size-1)/2);
figure
plot(vx)
title('Variance of deseasonalized output')

clear window_size,clear Cmx, clear Qx

%% NAN for deseasonalized variables
% Train
nan_idx=isnan(u);
sum(nan_idx);
dim_u = size(u);

% Substitute Nan value with the mean of the column
m_u = mean(u,1,'omitnan'); % It should be approximately zero (deseasonalized)
for i = 1:dim_u(1) % righe
    for h = 1:dim_u(2) % colonne
        if nan_idx(i,h) == 1
            u(i,h) = u(i-1,h); % m_u(h);
        end
    end
end

% Test
nan_idx=isnan(u_test);
sum(nan_idx);
dim_utest = size(u_test);

% Substitute Nan value with the mean of the column
m_utest = mean(u_test,1,'omitnan');
for i = 1:dim_utest(1) % righe
    for h = 1:dim_utest(2) % colonne
        if nan_idx(i,h) == 1
            u_test(i,h) = u_test(i-1,h);% m_utest(h);
        end
    end
end

clear nan_idx, clear m_u*, clear i, clear h

%  ----------------------------------------------
%%   Train the linear model: K-means clustering
%  ----------------------------------------------

% ____________________________________________________
% OPERATOR SELECTION
% Selection of the input used for ARX
sel_input_red = [1 9 13 16 22];        % dry % PCA 1 2
sel_input_blue = [1 6 8 13 16 22];     % wet % PCA 1 2
k_input = [0]; % 0 refers to the output while 1-25 the corresponding inputs

% Selection of the variables used for K-means. Training and Validation must be
% the same
k_sel_train = [];
k_sel_test = [];

for i = k_input
    if i == 0
        k_sel_train = [k_sel_train x];
        k_sel_test = [k_sel_test x_test];
    end
    if not(i==0)
        k_sel_train = [k_sel_train u(:,i)];
        k_sel_test = [k_sel_test u_test(:,i)];
    end
end

% Training

% Computing centroids
[idx,C] = kmeans(k_sel_train,2);
% C = zeros(size(u,1),size(u,2),3);
% D = zeros(size)
if norm(C(1,:))>norm(C(2,:))    % Ascending order
    H = C;
    C(1,:) = H(2,:);
    C(2,:) = H(1,:);
    temp = idx;
    for h = 1:1:length(temp)
        if (temp(h) == 1)
            idx(h) = 2;
        else 
            idx(h) = 1;
        end
    end
    clear H, clear h, clear temp
end

jr=1;
jb=1;

figure    % Used also with PCA
for i=1:1:length(idx)
    if idx(i)==1
        plot3(u(i,1),u(i,22),x(i),'r.')   % Used also with PCA
        hold on
        r=1;
        for h = sel_input_red
            Cr(jr,r) = u(i,h);
            r=r+1;
        end
        clear h, clear r
        Dr(jr,1) = x(i);
        indicer(jr,1)=i;
        jr= jr+1;

    end
    if idx(i)==2
        plot3(u(i,1),u(i,22),x(i),'b.')   % Used also with PCA
        hold on
        l=1;
        for h = sel_input_blue
           Cb(jb,l) = u(i,h);
           l = l+1;
        end
        clear h, clear l
        Db(jb,1) = x(i);
        indiceb(jb,1)=i;
        jb=jb+1;
    end
end
hold off 
clear i, clear h

% Plot of the timeseries deseasonalized and the two clusters are
% highlighted
figure
hold on
for i=1:1:length(idx)
    if idx(i)==1
        plot(t(i),x(i),'r.')
    end

    if idx(i)==2
        plot(t(i),x(i),'b.')
    end
    if idx(i)==3
        plot(t(i),x(i),'g.')
    end
end
title('timeseries deseasonalized split up in dry (red) and wet (blue) clusters - Training')
xlabel('time [days]')
ylabel('inflow [m^3/s]')
hold off
jr=1;
jb=1;
jg=1;

clear i

% VALIDATION
% computing d1 and d2 
d1 = [k_sel_test]-C(1,:);
d2 = [k_sel_test]-C(2,:);
d1_square=0;
d2_square=0;
h=0;

disp('Number of Outputs+Inputs used for K-means clustering'),
for i = 1:1:size(C,2)
    d1_square = d1_square+d1(:,i).^2;
    d2_square = d2_square+d2(:,i).^2;
    h = h+1;
end
disp(h)
clear h

d1 = sqrt(d1_square);
d2 = sqrt(d2_square);
clear i

% Creating red and blue clusters

% figure    % Used with PCA
for i = 1:1:length(x_test)
    if d1(i)<d2(i)
%         plot3(u_test(i,1),u_test(i,22),x(i),'r.') % PCA
%         hold on   % PCA
        r = 1;
        for h =sel_input_red
            Crtest(jr,r) = u_test(i,h);
            r = r+1;
        end
        clear h, clear r
        Drtest(jr,1) = x_test(i);
        indicetestr(jr,1)= i;
        jr=jr+1;
        idxtest(i)=1;
    else
%         plot3(u_test(i,1),u_test(i,22),x_test(i),'b.')    % PCA
%         hold on   % PCA
        l = 1;
        for h =sel_input_blue
            Cbtest(jb,l) = u_test(i,h);
            l = l+1;
        end
        clear h, clear l
        Dbtest(jb,1) = x_test(i);
        indicetestb(jb,1)=i;
        jb=jb+1;
        idxtest(i)=2;
    end
end
clear i,clear jr,clear jb

figure
hold on
for i=1:1:length(x_test)
    if idxtest(i)==1
        plot(t(i),x_test(i),'r.')
    end

    if idxtest(i)==2
        plot(t(i),x_test(i),'b.')
    end
end
title('timeseries deseasonalized split up in dry (red) and wet (blue) clusters - Validation')
xlabel('time [days]')
ylabel('inflow [m^3/s]')
hold off
clear i

%% Recursive ARX - RED
k=3; % This is the lead time between the actual istant time and the predicted one. 
ARXmax = 50;    % maximum ARX order tested

% Initialization
R2_i_red = zeros(ARXmax,2); % first column --> Ein ; second column --> Eout;
Ein_i_red = zeros(ARXmax,1);
Eout_i_red = zeros(ARXmax,1);
% q=10;

for q = 1:1:ARXmax % i is the actual ARX model

    % training
    [Mprova_red,Yin,betaprova_red] = ARX(q,k,Dr,Cr);
    Y_hatprova_red = [Yin;Mprova_red*betaprova_red];
    n_hatprova_red = (Y_hatprova_red.*sigma(indicer))+m(indicer);
    R2_i_red(q,1)=1-sum((n(indicer(length(Yin)+1:end))-n_hatprova_red(length(Yin)+1:end)).^2)/sum((n(indicer(length(Yin)+1:end))-m(indicer(length(Yin)+1:end))).^2);
    Ein_i_red(q) = sum((n(indicer(length(Yin)+1:end))-n_hatprova_red(length(Yin)+1:end)).^2)/length(n(indicer(length(Yin)+1:end)));
    clear Yin

    % validation
     [Mtestprova_red,Yin,betatestprova_red] = ARX(q,k,Drtest,Crtest);
     Ytest_hatprova_red = [Yin;Mtestprova_red*betaprova_red];
     ntest_hatprova_red=(Ytest_hatprova_red.*sigma_test(indicetestr))+m_test(indicetestr);
     R2_i_red(q,2)=1-sum((n_test(indicetestr(length(Yin)+1:end))-ntest_hatprova_red(length(Yin)+1:end)).^2)/sum((n_test(indicetestr(length(Yin)+1:end))-m_test(indicetestr(length(Yin)+1:end))).^2);
     Eout_i_red(q) = sum((n_test(indicetestr(length(Yin)+1:end))-ntest_hatprova_red(length(Yin)+1:end)).^2)/length(n_test(indicetestr(length(Yin)+1:end)));
     clear Yin
     disp('ARX '),disp(q);
end

figure
hold on
plot(Ein_i_red,'b')
plot(Eout_i_red,'r','LineWidth',2)
legend('Ein','Eout')
grid
title('DATASET RED')
xlabel('ARX(i,i)')
ylabel('Error')
hold off
clear q

%% Recursive ARX - BLUE

R2_i_blue = zeros(ARXmax,2); % first column --> Ein ; second column --> Eout;
Ein_i_blue = zeros(ARXmax,1);
Eout_i_blue = zeros(ARXmax,1);

for q = 1:1:ARXmax % i is the actual ARX model

    % training
    [Mprova_blue,Yin,betaprova_blue] = ARX(q,k,Db,Cb);
    Y_hatprova_blue = [Yin;Mprova_blue*betaprova_blue];
    n_hatprova_blue = (Y_hatprova_blue.*sigma(indiceb))+m(indiceb);
    R2_i_blue(q,1)=1-sum((n(indiceb(length(Yin)+1:end))-n_hatprova_blue(length(Yin)+1:end)).^2)/sum((n(indiceb(length(Yin)+1:end))-m(indiceb(length(Yin)+1:end))).^2);
    Ein_i_blue(q) = sum((n(indiceb(length(Yin)+1:end))-n_hatprova_blue(length(Yin)+1:end)).^2)/length(n(indiceb(length(Yin)+1:end)));
    clear Yin
    % validation
     [Mtestprova_blue,Yin,betatestprova_blue] = ARX(q,k,Dbtest,Cbtest);
     Ytest_hatprova_blue = [Yin;Mtestprova_blue*betaprova_blue];
     ntest_hatprova_blue=(Ytest_hatprova_blue.*sigma_test(indicetestb))+m_test(indicetestb);
     R2_i_blue(q,2)=1-sum((n_test(indicetestb(length(Yin)+1:end))-ntest_hatprova_blue(length(Yin)+1:end)).^2)/sum((n_test(indicetestb(length(Yin)+1:end))-m_test(indicetestb(length(Yin)+1:end))).^2);
     Eout_i_blue(q) = sum((n_test(indicetestb(length(Yin)+1:end))-ntest_hatprova_blue(length(Yin)+1:end)).^2)/length(n_test(indicetestb(length(Yin)+1:end)));
     clear Yin
     disp('ARX '),disp(q);
end

figure
hold on
plot(Ein_i_blue,'b')
plot(Eout_i_blue,'r','LineWidth',2)
legend('Ein','Eout')
grid
title('Dataset BLUE')
xlabel('ARX(i,i)')
ylabel('Error')
hold off

clear ARXMax, clear q

%% CHOICE OF BEST ARX MODELS

% RED
for i = 1:length(Eout_i_red)
    if Eout_i_red(i)==min(Eout_i_red(k+1:end))
        disp('BEST RED ARX:'),disp(i)
        Best_red_arx = i;
    end
end

clear i

% BLUE
for i = 1:length(Eout_i_blue)
    if Eout_i_blue(i)==min(Eout_i_blue(k+1:end))
        disp('BEST BLUE ARX:'),disp(i)
        Best_blue_arx=i;
    end
end

clear i

%% RED - ARX(best)

q = Best_red_arx;

% training
[M_red,Yin,beta_red] = ARX(q,k,Dr,Cr);
Y_hat_red = [Yin;M_red*beta_red];
n_hat_red = (Y_hat_red.*sigma(indicer))+m(indicer);
R2_red(1,1)=1-sum((n(indicer)-n_hat_red(1:end)).^2)/sum((n(indicer)-m(indicer)).^2);
Ein_red = sum((n(indicer)-n_hat_red(1:end)).^2)/length(n(indicer));
clear Yin
% validation
 [Mtest_red,Yin,betatest_red] = ARX(q,k,Drtest,Crtest);
 Ytest_hat_red = [Yin;Mtest_red*beta_red];
 ntest_hat_red=(Ytest_hat_red.*sigma_test(indicetestr))+m_test(indicetestr);
 R2_red(1,2)=1-sum((n_test(indicetestr(length(Yin)+1:end))-ntest_hat_red(length(Yin)+1:end)).^2)/sum((n_test(indicetestr(length(Yin)+1:end))-m_test(indicetestr(length(Yin)+1:end))).^2);
 Eout_red = sum((n_test(indicetestr(length(Yin)+1:end))-ntest_hat_red(length(Yin)+1:end)).^2)/length(n_test(indicetestr(length(Yin)+1:end)));
 clear Yin
 disp('RED')
 disp('ARX '),disp(q)
 disp('R2'),disp(R2_red)
 disp('Ein'),disp(Ein_red)
 disp('Eout'),disp(Eout_red)
clear q
 %% BLUE - ARX(best)

q =  Best_blue_arx;

% training
[M_blue,Yin,beta_blue] = ARX(q,k,Db,Cb);
Y_hat_blue = [Yin;M_blue*beta_blue];
n_hat_blue = (Y_hat_blue.*sigma(indiceb))+m(indiceb);
R2_blue(1,1)=1-sum((n(indiceb(length(Yin)+1:end))-n_hat_blue(length(Yin)+1:end)).^2)/sum((n(indiceb(length(Yin)+1:end))-m(indiceb(length(Yin)+1:end))).^2);
Ein_blue = sum((n(indiceb(length(Yin)+1:end))-n_hat_blue(length(Yin)+1:end)).^2)/length(n(indiceb(length(Yin)+1:end)));
clear Yin
% validation
 [Mtest_blue,Yin,betatest_blue] = ARX(q,k,Dbtest,Cbtest);
 Ytest_hat_blue = [Yin;Mtest_blue*beta_blue];
 ntest_hat_blue=(Ytest_hat_blue.*sigma_test(indicetestb))+m_test(indicetestb);
 
 R2_blue(1,2)=1-sum((n_test(indicetestb(length(Yin)+1:end))-ntest_hat_blue(length(Yin)+1:end)).^2)/sum((n_test(indicetestb(length(Yin)+1:end))-m_test(indicetestb(length(Yin)+1:end))).^2);
 Eout_blue = sum((n_test(indicetestb(length(Yin)+1:end))-ntest_hat_blue(length(Yin)+1:end)).^2)/length(n_test(indicetestb(length(Yin)+1:end)));
 clear Yin
 disp('BLUE')
 disp('ARX '),disp(q)
 disp('R2'),disp(R2_blue)
 disp('Ein'),disp(Ein_blue)
 disp('Eout'),disp(Eout_blue)
clear q

%% RED + BLUE reconstruction

% Training
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
        disp('A problem in reconstructing training set has occured!')
    end
end
clear i, clear a

% Validation
jr = 1;
jb = 1;
for i = 1:1:length(idxtest)
    if idxtest(i)==1
        ntest_hat(i,1) = ntest_hat_red(jr);
        jr = jr+1;
        a(i) = 1; % to check if all values are 1
    end
    if idxtest(i)==2
        ntest_hat(i,1) = ntest_hat_blue(jb);
        jb = jb+1;
        a(i) = 1;
    end
end
clear i, clear jr, clear jb

for i = 1:1:length(a) 
    if not(a(i)==1)
        disp('A problem in reconstructing validation set has occured!')
    end
end
clear i, clear a

%% Analysis of Prediction
R2(1,1) = 1-sum((n(1:end)-n_hat(1:end)).^2)/sum((n(1:end)-m(1:end)).^2);
R2(1,2) = 1-sum((n_test(1:end)-ntest_hat(1:end)).^2)/sum((n_test(1:end)-m_test(1:end)).^2);
disp('R2'),disp(R2)

figure
hold on
plot(t,n,'b')
plot(t,n_hat,'r')
legend('n','n hat')
title('trainging - actual timeseries vs forecasted one')
xlabel('time [days]')
ylabel('inflow [m^3/s]')
hold off

j = 1:length(n_test);
j=j';
figure
hold on
plot(j,n_test,'b')
plot(j,ntest_hat,'r')
legend('n test','ntest hat')
title('validation - actual timeseries vs forecasted one')
xlabel('time [days]')
ylabel('inflow [m^3/s]')
hold off
figure
hold on
plot([t; t(end)+j],[n; n_test],'b')
plot([t; t(end)+j],[n_hat; ntest_hat],'r')
legend('n test','ntest hat')
title('ALL timeseries (training+validation)')
xlabel('time [days]')
ylabel('inflow [m^3/s]')
hold off
clear j

% Training
Residuals = n(1:end)-n_hat(1:end);
figure
correlogram(Residuals,Residuals,5)
title('Correlogram residual MA(2) process')
figure
correlogram(Residuals,n_hat,5)
title('Correlation between residual and prediction TRAINING')

% Validation
Residuals = n_test(1:end)-ntest_hat(1:end);
figure
correlogram(Residuals,Residuals,5)
title('Correlogram residual MA(2) process')
figure
correlogram(Residuals,ntest_hat,5)
title('Correlation between residual and prediction VALIDATION')
%  --------------------
%%   NON LINEAR MODEL
%  --------------------
% to skip this part set NNswitch=0 at the beginning; otherwise = 1
if (NNswitch==1)
    [M_ann_cal, Y_ann_cal]=generate_M(u, x, 1);
    [M_ann_val, Y_ann_val]=generate_M(u_test, x_test, 1);
    [ann_1, R2_1]=rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, n_test, m_test, sigma_test, 1);
    disp('1')
   
    [M_ann_cal, Y_ann_cal]=generate_M(u, x, 2);
    [M_ann_val, Y_ann_val]=generate_M(u_test, x_test, 2);
    [ann_2, R2_2]=rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, n_test, m_test, sigma_test, 2);
    disp('2')

    [M_ann_cal, Y_ann_cal]=generate_M(u, x, 3);
    [M_ann_val, Y_ann_val]=generate_M(u_test, x_test, 3);
    [ann_3, R2_3]=rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, n_test, m_test, sigma_test, 3);
    disp('3')

    [M_ann_cal, Y_ann_cal]=generate_M(u, x, 4);
    [M_ann_val, Y_ann_val]=generate_M(u_test, x_test, 4);
    [ann_4, R2_4]=rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, n_test, m_test, sigma_test, 4);
    disp('4')

    [M_ann_cal, Y_ann_cal]=generate_M(u, x, 5);
    [M_ann_val, Y_ann_val]=generate_M(u_test, x_test, 5);
    [ann_5, R2_5]=rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, n_test, m_test, sigma_test, 5);
    disp('5')
   
end

if (NNswitch==1)
    %NON linear part
    R2_ann=[R2_1 R2_2 R2_3 R2_4 R2_5];
    figure
    plot (R2_ann, 'o')
    title ("R2 in test for ANN models")
    xlabel ("Number of x used")
    ylabel("R2")
    
    %The best is the ANN(2)
    best_ann=ann_2;
end
%  --------------------
%%   Save best model
%  --------------------

SavedValues = C;
Dim = size(C);
[SavedValues,Dim] = InsertParam(beta_red,SavedValues,Dim);
[SavedValues,Dim] = InsertParam(beta_blue,SavedValues,Dim);
[SavedValues,Dim] = InsertParam(Best_red_arx,SavedValues,Dim);
[SavedValues,Dim] = InsertParam(Best_blue_arx,SavedValues,Dim);
[SavedValues,Dim] = InsertParam(sel_input_red',SavedValues,Dim); 
[SavedValues,Dim] = InsertParam(sel_input_blue',SavedValues,Dim);
len = size(Dim,1);
[SavedValues,~] = InsertParam(Dim,SavedValues,Dim);
[SavedValues,~] = InsertParam(len,SavedValues,Dim);


save Parametri.txt SavedValues -ascii
clear SavedValues, clear Dim