off102 = csvread('ch0res_2isol_offset102');
off102x = real(off102); off102y = imag(off102);
off202 = csvread('ch0res_2isol_offset202');
off202x = real(off202); off202y = imag(off202);
off302 = csvread('ch0res_2isol_offset302');
off302x = real(off302); off302y = imag(off302);
##off402 = csvread('ch0res_2isol_offset402');
##off402x = real(off402); off402y = imag(off402);
plot(actualx, actualy, off102x, off102y, off202x, off202y, off302x, off302y);#, off402x, off402y);
legend(['actual';'100';'200';'300'
        #;'400'
        ]
        , 'fontsize', 14)
