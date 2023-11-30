function [s1,r1] = massBalance( s, u, q )

% Output:
%      s1 - final storage. 
%      r1 - release over the 24 hours. 
%
% Input:
%       s - initial storage. 
%       u - release decision 
%       q - inflow 
%
% See also MIN_RELEASE, MAX_RELEASE

global sys_param;

HH = 24;
delta = sys_param.delta/HH;
s_ = nan(HH+1,1);
r_ = nan(HH+1,1);

s_(1) = s;
for i=1:HH
  qm = min_release(s_(i));
  qM = max_release(s_(i));
  r_(i+1) = min( qM , max( qm , u ) );
  s_(i+1) = s_(i) + delta*( q - r_(i+1) );
end

s1 = s_(HH);
r1 = mean(r_(2:end));
