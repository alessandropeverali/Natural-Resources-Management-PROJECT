function [ h, u, r ] = simComo( qq, h_in, theta )

%   OUTPUT
%          h - time-series of lake level. ([m])
%          u - time-series of release decision. ([m3/s])
%          r - time-series of actual release. ([m3/s])

%   INPUT
%          q - inflow time series. ([m3/s])
%          h_in - initial lake level. ([m])
%          policy - structure variable containing the policy related parameters.

global sys_param;
% Simulation setting
q_sim = [ nan; qq.q_Como ];
H = length(q_sim) - 1;

% Pre-allocation
[h,s,r,u] = deal(nan(size(q_sim)));

% Initialization
h(1) = h_in;
s(1) = (h(1) - sys_param.h0)*sys_param.S;

for t = 1: H
  
  % Compute release decision
  u(t) = std_operating_policy(h(t),theta);
  
  % Hourly integration of mass-balance equation
  [s(t+1), r(t+1)] = massBalance( s(t), u(t), q_sim(t+1) );
  h(t+1) = s(t+1) / sys_param.S + sys_param.h0 ;
  
end

end