function gt_low = g_low_level(h)

global sys_param ;

% low level
threshold = sys_param.h_low ; 

gt_low=(h<=threshold);
end