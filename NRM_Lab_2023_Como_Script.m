close all
clear
clc

%% Data import and model setup

global sys_param;
%Select simulation time interval
qq = model_setup(2009, 2019); % structure containing trajectories of inflow and parameters

%% Simulation of current policy

% Set SDP policy parameters
%theta = [0, 0.6, 530, 4900, 175, 0]; % h1, h2, m1, m2, c, m3
%I transform the parameters into slope and intercept
slopes=[530, 4900, 0];
intercept=[0, 0, 0];
const=175;
h1=0;
h2=0.6;
intercept(1)=const;
intercept(3)=const;
intercept(2)=const-slopes(2)*h2;
theta=[slopes, intercept];
clear lopes
clear intercept
clear h*
clear const
% Visualize policy
hh = -0.6:.01:1.5 ;
rmin=ones(size(hh));
rmax=rmin;
ss=(hh-sys_param.h0)*sys_param.S;%Real storage
for i = 1:length(hh)
    rmin(i) = min_release(ss(i));%Min possible release with that storage
    rmax(i) = max_release(ss(i));%Max possible release with that storage
end
 rmax=max(rmax, rmin);
 uu = std_operating_policy(hh, theta);%What I want to release
 rr = max(rmin, min(rmax, uu));%What I am able to release

figure;
hold on
siz=2;
plot(hh,rmin, '.-.','LineWidth', siz)
plot (hh,rr, 'r','LineWidth', siz)
plot (hh,rmax, '--','LineWidth', siz)
legend('min release', 'real release', 'max release', ...
    'location', 'north', 'FontSize', 18)
xlabel("level[m]")
ylabel("release [m^3/s]")
title ("Actual operating policy")
%% Plot of w
figure
plot (sys_param.w)
title ("water demand")
xlabel("days")
ylabel("m^3/s")
clear i
clear siz
%%
% set initial condition and run simulation
global h_init;
h_init=0.71;
[h, u, r]=simComo(qq, h_init, theta);
q=[nan;qq.q_Como];

% visualize trajectories
date_day = datetime(2009,1,1):datetime(2019,12,31);
date_day = date_day(~(day(date_day) == 29 & month(date_day) ==2));
figure; 
subplot(311); plot (date_day, q(2:end)); title("Inflow", 'FontSize',13); ylabel("m^3/s")%Inflow
subplot(312); plot (date_day, h(2:end)); title ("Level", 'FontSize',13); ylabel("m")%Level
subplot(313); plot (date_day, r(2:end)); title("Release", 'FontSize',13); ylabel("m^3/s")%Release

%% compute policy perfomance
gt_flo = g_flood( h(2:end) );
gt_def = g_deficit( r(2:end) );
gt_low = g_low_level( h(2:end));

n_years = length(h(2:end))/365;
Jflo = sum(gt_flo)/n_years;
Jdef = sum(gt_def)/(n_years*365);
Jlow = sum(gt_low)/n_years;
clear gt*

%% preparation for NSGAII

global opt_inputs ;
opt_inputs.qq = qq;
opt_inputs.h0 = sys_param.h0 ;

pop = 140; % number of individuals in the population
gen = 5; % number of generations     
M = 3; % number of objectives      
V = 6; % number of decision variables (policy parameters)
min_range = [0,0,0,-5000, -5000, -5000]; % minimum value of each parameter
max_range = [10000, 10000, 10000, 500, 500, 500]; % maximum value of each parameter
%% real NSGAII
[ chr0, chrF ] = nsga_2(pop,gen,M,V,min_range,max_range) ; 

%% standardization of performances
obj_F=chrF(:,V+1:V+M);
std_F=obj_F;
std_act=[Jflo, Jdef, Jlow];
g_min=[min(obj_F(:,1)), min(obj_F(:,2)), min(obj_F(:,3))];
g_max=[max(obj_F(:,1)), max(obj_F(:,2)), max(obj_F(:,3))];
for i=1:3
    std_F(:,i)=(std_F(:,i)-g_min(i))/(g_max(i)-g_min(i));
    std_act(:,i)=(std_act(:,i)-g_min(i))/(g_max(i)-g_min(i));
end

%% Selection of alternatives
%The best 3, for flood deficit and low low level prevention
best_flo_id=min(find(obj_F(:,1)==min(obj_F(:,1))));
best_def_id=min(find(obj_F(:,2)==min(obj_F(:,2))));
best_low_id=min(find(obj_F(:,3)==min(obj_F(:,3))));
best_flo_alt=chrF(best_flo_id, 1:V);
best_def_alt=chrF(best_def_id, 1:V);
best_low_alt=chrF(best_low_id, 1:V);

%The one with the smallest standardized distance from the origin
mean_obj=sum(std_F.^2, 2);
comp_id=find(mean_obj==min(mean_obj));
compr_alt=chrF(comp_id,1:V);

%% savage of the most usefull variables
sol = chrF (:,1:V+M);
save group.mat sol

%% Plot standard operating policy for the 4 models
hh = -0.6:.01:1.5 ;
ss=(hh-sys_param.h0)*sys_param.S;%Real storage
for i=1:length(hh)
    rmin(i) = min_release(ss(i));%Min possible release with that storage
    rmax(i) = max_release(ss(i));%Max possible release with that storage
end
rmax=max(rmax, rmin);
u_def = std_operating_policy(hh, best_def_alt);
r_def = max(rmin, min(rmax, u_def));
u_flo = std_operating_policy(hh, best_flo_alt);
r_flo = max(rmin, min(rmax, u_flo));
u_low = std_operating_policy(hh, best_low_alt);
r_low = max(rmin, min(rmax, u_low));
u_comp = std_operating_policy(hh, compr_alt);
r_comp = max(rmin, min(rmax, u_comp));
clear u_*
figure;
hold on
plot (hh, rr, 'r', 'LineWidth',3)
plot (hh,r_def, 'g', 'LineWidth',3)
plot (hh,r_flo, 'b', 'LineWidth',3)
plot (hh,r_low, 'y', 'LineWidth',3)
plot (hh,r_comp, 'k', 'LineWidth',3)
xlabel("level[m]")
ylabel("release [m^3/s]")
legend ('actual policy', 'best def', 'best flood', ...
    'best low level', 'compromise', 'location', 'north', 'FontSize', 15)
title ("chosen operating policies")

%% 3D plot of performances
obj_0=chr0(:,V+1:V+M);
obj_F=chrF(:,V+1:V+M);
ch_alt=[(best_def_id),(best_low_id),(best_flo_id),(comp_id)];
best_alt=obj_F(ch_alt,:);
figure
plot3(obj_F(:, 1), obj_F(:, 2), obj_F(:, 3), 'r+', ...
    'MarkerSize',13)

hold on
grid on
plot3(obj_0(:, 1), obj_0(:, 2), obj_0(:, 3), 'r.','MarkerSize',13)
plot3(Jflo, Jdef, Jlow, 'ko', 'MarkerSize',13, 'MarkerFaceColor', 'red')
plot3(best_alt(1,1), best_alt(1,2), best_alt(1,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'green')
plot3(best_alt(2,1), best_alt(2,2), best_alt(2,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'blue')
plot3(best_alt(3,1), best_alt(3,2), best_alt(3,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'yellow')
plot3(best_alt(4,1), best_alt(4,2), best_alt(4,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'black')
legend ('last gen', 'first gen', 'actual policy',...
    'best for deficit', 'best for low levels', 'best for flood', ...
    'compromise','location', 'best')
xlabel('x=flood control')
ylabel('y=deficit')
zlabel('z=low level prevention')
title ('3D plot of performances')
%% 3D prot of standardized
best_alt=std_F(ch_alt,:);
plot3(std_F(:, 1), std_F(:, 2), std_F(:, 3), 'm+')
hold on
grid on
plot3(std_act(1), std_act(2), std_act(3), 'ko', 'MarkerSize',13, ...
    'MarkerFaceColor', 'red')
plot3(best_alt(1,1), best_alt(1,2), best_alt(1,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'green')
plot3(best_alt(2,1), best_alt(2,2), best_alt(2,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'blue')
plot3(best_alt(3,1), best_alt(3,2), best_alt(3,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'yellow')
plot3(best_alt(4,1), best_alt(4,2), best_alt(4,3), 'kO', ...
    'MarkerSize',15, 'MarkerFaceColor', 'black')
legend ('last gen', 'actual policy',...
    'best for deficit', 'best for low levels', 'best for flood', ...
    'compromise','location', 'best')
xlabel('x=flood control')
ylabel('y=deficit')
zlabel('z=low level prevention')
title ('plot of standardized performances of the last generation')