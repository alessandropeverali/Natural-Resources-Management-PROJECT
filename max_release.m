function V = max_release(s)


global sys_param;

h = s / sys_param.S + sys_param.h0 ;
V = 0.0;
    
idx = 33.37*(sys_param.h0 + 0.1 + 2.5).^2.015;
m = idx/0.1;
it = -(idx/0.1)*sys_param.h0;
    
if h <= sys_param.h0
    V = 0.0;
elseif h <= sys_param.h0 + 0.1
    V = m*h + it;
else
    V = 33.37*(h + 2.5).^2.015 ;
end


end