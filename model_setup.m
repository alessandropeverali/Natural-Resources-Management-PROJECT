function qq = model_setup(ini_year, fin_year)
%date structure = [YYYY]

global sys_param ;

% Load inflow and demand
X_Como  = load('Lake_Como_Data_2009_2019.txt','-ascii') ; % "day" "month" "year" "level" "release" "inflow"
w   = load('aggregated_demand.txt','-ascii') ;

% Select time horizon
idx_in_year = X_Como(:,3)>(ini_year-1);
X_in_year = X_Como(idx_in_year,:);
idx_fin_year = X_in_year(:,3)<(fin_year+1);
X_fin_year = X_in_year(idx_fin_year,end);
X_sim = X_fin_year;

% remove 29/feb
dn = datenum(ini_year,1,1):datenum(fin_year,12,31);
dv = datevec(dn');
feb29 = dv(:,2).*dv(:,3) == 2*29;
X_sim = X_sim(~feb29,:) ;
qq.q_Como = X_sim(:,end);

% Downstream model parameters:
sys_param.h_flo = 1.1;  % flood threshold 
sys_param.h1 = 1.1;    %upper bound of the operating space
sys_param.h_low = -0.20; % low level threshold
sys_param.MEF = 22;     % minimum environmental flow
sys_param.h0 = -0.40;   % lower bound of the operating space
sys_param.S = 145900000 ; % lake surface
sys_param.T = 365 ;     % the period is equal 1 year
sys_param.delta = 60*60*24;    % daily time-step
sys_param.w = w;   % total demand (irrigation + HP)