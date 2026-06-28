clc;
close all;

X=[-800.000 -100  -80.0 -60 -50.0 -40 -30.0 -20 -10  0.00000 10  20 30.0000 40  50.0000  60 80 100.0000 125.0  150.00 200.0  800.000];
Y=[-800.000 -225.0  -175.0 -125.0 -110 -100 -90 -80 -70 -60  -50.0 -40.0 -30 -20 -10 0.00000 10  20  30.0000   50.0000  75.0  100.000  150.00  225.0  800.000];
Z=[-5.00000   0.00000   5.0000   10.0  15.0000 20.0  25.0000  30.0  35.000   45.  100.0];



nx=22;
ny=24;
nz=13;

ColorJet=colormap('jet');
ColorNJet=flipud(ColorJet);
red2blue=load('~/matlab/red2blue');
use=red2blue;

% read velocity data
% read velocity data
vel1=load('10km.gap10percFine.Vp_model.dat');
%vel1=load('alternate.gap10percFine.Vp_model.dat');
%vel1=load('MOD.10checkerPS.fine2.nohead');
%vel1=load('neicfake.gap10percFine.Vp_model.dat');
%vel1=vel1./vel2;%
%vel0=load('MOD.solomon1DPS.fine.nohead');
vel0=load('MOD.solomon1DP.fine.nohead');
%diffvel=vel1;
diffvel=100*((vel1-vel0)./vel0);
%loc=load('10km.gap10percFine.tomoDD.loc');
loc=load('neicfake.gap10percFine.tomoDD.loc');
sta=load('solomon.sta.car');
% dws=load('dws_fix_p');

VEL=[];
for k=1:nz-2
    for i=2:ny-1
        for j=2:nx-1
            VEL(j-1,i-1,k)=diffvel(k*ny+i,j);
            %DWS(j-1,i-1,k)=dws(k*ny+i,j);
        end
    end
end

for i=1:length(loc)
    lat(i)=loc(i,2);
    lon(i)=loc(i,3);
    y(i)=loc(i,6)/1000;
    x(i)=loc(i,5)/1000;
    z(i)=loc(i,4);
end

%Some defaults

velcontour=[-10 -8 -6 -4 -2 0 2 4 6 8 10];
crossh=[];
%crossdws=[];
m=1;
%Loop over depth
for j=12:2:16
    for i=2:ny-1
        for k=1:nz-2
            crossh(k,i-1)=VEL(j-1,i-1,k);
            %crossdws(k,i-1)=DWS(j-1,i-1,k);
        end
    end
    subplot(3,1,m), caxis([-10,10]); hold on;
    subplot(3,1,m), caxis manual; hold on;
    colormap(ColorNJet)	
    subplot(3,1,m), pcolor(Y(2:ny-1),Z(1:nz-2),crossh); hold on;
    shading interp;
    for i=2:ny-1
    	subplot(3,1,m), plot(Y(i),Z,'xm','markersize',3); hold on;
    end
    subplot(3,1,m), [c,h]=contour(Y(2:ny-1),Z(1:nz-2),crossh,velcontour,'k'); hold on;
    %subplot(3,1,m), [cd,hd]=contour(Y(2:ny-1),Z(1:nz-2),crossdws,dwscontour,'w'); hold on;
    subplot(3,1,m), clabel(c,h);
    %clabel(cd,hd);
   
    for l=1:length(loc)
	if x(l)>(X(j)-(X(j-1))/2) && x(l)<=(X(j)+((X(j+1)-X(j))/2))
           subplot(3,1,m), plot(y(l),z(l),'ok','markersize',3);
        end
    end 
    title(['X=',num2str(X(j)),' km']); 
    xlabel('Y(km)'); 
    ylabel('Z(km)'); 
    axis image;
    axis ij;
    axis([-50,90,Z(2),Z(nz-2)]);
    colorbar;   
    hold off;
 
    m=m+1;
end


