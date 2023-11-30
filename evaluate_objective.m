function f = evaluate_objective(x, M, V)
%
% function f = evaluate_objective(x, M, V)
%
% Function to evaluate the objective functions for the given input vector x.%
% x is an array of decision variables and f(1), f(2), etc are the
% objective functions. The algorithm always minimizes the objectivehjghl

% function hence if you would like to maximize the function then multiply
% the function by negative one. M is the numebr of objective functions and
% V is the number of decision variables. 
%
% This functions is basically written by the user who defines his/her own
% objective function. Make sure that the M and V matches your initial user
% input.
%

x = x(1:V) ;
x = x(:)   ;

% --------------------------------------
% insert here your function:
% global variable to pass inside extra inputs
global opt_inputs ;
global sys_param ;
qq = opt_inputs.qq ;
h0 = opt_inputs.h0 ;
% 1) policy param
theta = x';
% 2) run simulation
global h_init;

[h, u, r]=simComo(qq, h_init, theta);
q=[nan;qq.q_Como];

% 3) compute objs
gt_flo = g_flood( h(2:end) ); 
gt_def = g_deficit( r(2:end) ); 
gt_low = g_low_level( h(2:end)); 

n_years = length(h(2:end))/365;
Jflo = sum(gt_flo)/n_years;
Jdef = sum(gt_def)/(n_years*365);
Jlow = sum(gt_low)/n_years;

f = [ Jflo, Jdef, Jlow ];
% --------------------------------------

% Check for error
if length(f) ~= M
    error('The number of decision variables does not match you previous input. Kindly check your objective function');
end