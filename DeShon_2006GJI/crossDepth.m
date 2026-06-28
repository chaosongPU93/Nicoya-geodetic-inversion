clc
figure
%close all;

X=[-395.000 -140.0 -100.0 -60.0 -40.0 -20.0 -10.0 0.0 10.0 20.0 25.0 30.0 35.0 40.0 45.0 50.0 55.0 60.0 70.0 80.0 115.0 150.0 595.000];
Y=[-395.000 -140.0 -100.0 -80.0 -60.0 -40.0 -20.0 -10.0 0.0 5.0 10.0 15.0 20.0 25.0 30.0 35.0 40.0 50.0 60.0 80.0 110.0 140.0 595.000];
%Z=[-10.0  0.  1.0  3.0  5.0 9.0  12.0 15.0 20.0 25.0 35.0];
Z=[-10.0  0.  1.0  2.0  4.0 6.0  9.0  12.0 15.0 20.0 30.0 40.0];

nx=23;
ny=23;
%nz=13;
nz=14;

ColorJet=colormap('jet');
ColorNJet=flipud(ColorJet);
red2blue=load('/Users/hrdeshon/matlab/red2blue');

% read velocity data
%inP=load('1D.MOD');
%vpvs=load('1D.VpVs.MOD');
inP=load('hrd_model/1D.MOD');
vpvs=load('hrd_model/1D.VpVs.MOD');

inS=inP./vpvs;

%velP=load('all_PS.Vp_model.dat.fixed');
%velS=load('all_PS.Vs_model.dat.fixed');

velP=load('hrd_model/hrd.all_PS.Vp_model.dat.fixed');
velS=load('hrd_model/hrd.all_PS.Vs_model.dat.fixed');

velPS=velP./velS;
%vel=load('meredith.Vp_model.dat');
%velP=load('MOD_lowfaults_Vp.dat');
velP=100.*((velP-inP)./inP);
%velS=load('all_PS.Vs_model.dat.fixed');
%velS=load('MOD_lowfaults_Vs.dat');
velS=100.*((velS-inS)./inS);
loc=load('all_PS.reloc');
loc=load('hrd_model/hrd.all_PS.reloc');
dwsP=load('dws_fix_p');
dwsS=load('dws_fix_s');

VELP=[];VELS=[];VELPS=[];
for k=1:nz-2
    for i=2:ny-1
        for j=2:nx-1
            VELP(j-1,i-1,k)=velP(k*ny+i,j);
            VELS(j-1,i-1,k)=velS(k*ny+i,j);
            VELPS(j-1,i-1,k)=velPS(k*ny+i,j);
            DWSP(j-1,i-1,k)=dwsP(k*ny+i,j);
            DWSS(j-1,i-1,k)=dwsS(k*ny+i,j);
       
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
dwscontour=[50 250];
%velcontour=[-10 0 10];
crosshP=[];crosshS=[];crosshPS=[];crossdwsP=[];crossdwsS=[];
m=1;
%Loop over depth
for k=4:7
    for i=2:ny-1
        for j=2:nx-1
            crosshP(i-1,j-1)=VELP(j-1,i-1,k);
            crosshS(i-1,j-1)=VELS(j-1,i-1,k);
            crosshPS(i-1,j-1)=VELPS(j-1,i-1,k);
            crossdwsP(i-1,j-1)=DWSP(j-1,i-1,k);
            crossdwsS(i-1,j-1)=DWSS(j-1,i-1,k);
        end
    end
    
    subplot(4,4,m), caxis([-6,6]); hold on;
    subplot(4,4,m), caxis manual; hold on;
	colormap(ColorNJet)
    subplot(4,4,m), pcolor(X(2:nx-1),Y(2:ny-1),crosshP); hold on;
    shading interp;
    for j=2:nx-1
    	subplot(4,4,m), plot(X(j),Y,'xm','markersize',3); hold on;
    end
  %  subplot(3,2,m), [c,h]=contour(X(2:nx-1),Y(2:ny-1),crossh,velcontour,'k'); hold on;
    subplot(4,4,m), [cd,hd]=contour(X(2:nx-1),Y(2:ny-1),crossdwsP,dwscontour,'k'); hold on;
  %  subplot(3,2,m), clabel(c,h);
    %clabel(cd,hd);
   
    for l=1:length(loc)
        if (k-1)==0, top=-10;, else top=Z(k)-((Z(k)-Z(k-1))/2);, end
        if (k+1)==nz-1, bottom=60;, else bottom=Z(k)+((Z(k+1)-Z(k))/2);, end
        if z(l)>=top && z(l)<bottom
           subplot(4,4,m), plot(x(l),y(l),'ok','markersize',3);
        end
    end 
    title(['Z=',int2str(Z(k)),' km']); 
    xlabel('X(km)'); 
    ylabel('Y(km)'); 
    axis image;
    %axis([X(5),X(nx-3),Y(5),Y(ny-3)]);
    axis([-45,90,-50,90]);
    %colorbar;   
    hold off;
 
    
    subplot(4,4,m+1), caxis([-6,6]); hold on;
    subplot(4,4,m+1), caxis manual; hold on;
	colormap(ColorNJet)
    subplot(4,4,m+1), pcolor(X(2:nx-1),Y(2:ny-1),crosshS); hold on;
    shading interp;
    for j=2:nx-1
    	subplot(4,4,m+1), plot(X(j),Y,'xm','markersize',3); hold on;
    end
  %  subplot(3,2,m), [c,h]=contour(X(2:nx-1),Y(2:ny-1),crossh,velcontour,'k'); hold on;
    subplot(4,4,m+1), [cd,hd]=contour(X(2:nx-1),Y(2:ny-1),crossdwsS,dwscontour,'k'); hold on;
  %  subplot(3,2,m), clabel(c,h);
    %clabel(cd,hd);
   
    for l=1:length(loc)
        if (k-1)==0, top=-10;, else top=Z(k)-((Z(k)-Z(k-1))/2);, end
        if (k+1)==nz-1, bottom=60;, else bottom=Z(k)+((Z(k+1)-Z(k))/2);, end
        if z(l)>=top && z(l)<bottom
           subplot(4,4,m+1), plot(x(l),y(l),'ok','markersize',3);
        end
    end 
    title(['Z=',int2str(Z(k)),' km']); 
    xlabel('X(km)'); 
    ylabel('Y(km)'); 
    axis image;
    %axis([X(5),X(nx-3),Y(5),Y(ny-3)]);
    axis([-45,90,-50,90]);
    %colorbar;   
    hold off;
    
    subplot(4,4,m+2), caxis([1.6,1.86]); hold on;
    subplot(4,4,m+2), caxis manual; hold on;
	colormap(ColorNJet)
    subplot(4,4,m+2), pcolor(X(2:nx-1),Y(2:ny-1),crosshPS); hold on;
    shading interp;
    for j=2:nx-1
    	subplot(4,4,m+2), plot(X(j),Y,'xm','markersize',3); hold on;
    end
  %  subplot(3,2,m), [c,h]=contour(X(2:nx-1),Y(2:ny-1),crossh,velcontour,'k'); hold on;
    subplot(4,4,m+2), [cd,hd]=contour(X(2:nx-1),Y(2:ny-1),crossdwsS,dwscontour,'k'); hold on;
  %  subplot(3,2,m), clabel(c,h);
    %clabel(cd,hd);
   
    for l=1:length(loc)
        if (k-1)==0, top=-10;, else top=Z(k)-((Z(k)-Z(k-1))/2);, end
        if (k+1)==nz-1, bottom=60;, else bottom=Z(k)+((Z(k+1)-Z(k))/2);, end
        if z(l)>=top && z(l)<bottom
           subplot(4,4,m+2), plot(x(l),y(l),'ok','markersize',3);
        end
    end 
    title(['Z=',int2str(Z(k)),' km']); 
    xlabel('X(km)'); 
    ylabel('Y(km)'); 
    axis image;
    %axis([X(5),X(nx-3),Y(5),Y(ny-3)]);
    axis([-45,90,-50,90]);
    %colorbar;   
    hold off;
    
    latlim=[35.75 37];
    lonlim=[-90.5 -89];
    subplot(4,4,m+3), ax = usamap(latlim,lonlim); hold on;
    states = shaperead('usastatehi',...
    'UseGeoCoords', true, 'BoundingBox', [lonlim', latlim']);
    subplot(4,4,m+3), geoshow(ax,states,'facecolor', 'w','edgecolor','k'); hold on;
    axis on;
    hold off;

    m=m+4;
end

hold off;
