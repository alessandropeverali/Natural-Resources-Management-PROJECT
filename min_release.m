function v = min_release(s)

global sys_param;

h = s / sys_param.S + sys_param.h0 ;

v = 0.0;

if h <= sys_param.h0
    v = 0.0;
elseif h <= sys_param.h1
    v = sys_param.MEF ;
else
    v = 33.37*(h + 2.5).^2.015 ;
end

end