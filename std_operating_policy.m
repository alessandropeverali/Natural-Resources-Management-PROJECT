function u = std_operating_policy(h, theta)

global sys_param
% -- Get policy parameters --
m1=theta(1);
q1=theta(4);
m2=theta(2);
q2=theta(5);
m3=theta(3);
q3=theta(6);
% -- Construct the policy using piecewise linear functions --
    L1=m1*h+q1;
    L2=m2*h+q2;
    L3=m3*h+q3;
    u=max(min(L3, L1), L2);

% -- Verify release are non-negative  --
u=max(0, u);

end