function gt_flo = g_flood(h)

global sys_param ;
% level excess

threshold = sys_param.h_flo ;
gt_flo=(h>threshold);


end
