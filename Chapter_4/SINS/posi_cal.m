
function [posiN]=posi_cal(T,veloN,posiN)

posiN = posiN + veloN * T;