function gt_def = g_deficit(r)

global sys_param ;

w = sys_param.w ;
powers=ones(size(r));
T=365;
n=round(length(r)/T);
w=repmat(w, n, 1);

for i=1:n
    powers((T*(i-1)+91):(T*(i-1)+283))=2;
end
deficit=max(0, w-r);

gt_def=deficit.^powers;

end
