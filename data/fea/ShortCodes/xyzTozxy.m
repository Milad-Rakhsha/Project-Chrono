clear
clc
syms cx cy cz sx sy sz
% z=[cz -sz 0; sz cz 0; 0 0 1]
% y=[cy 0 -sy; 0 1 0; sy 0 cy]
% x=[1 0 0; 0 cx -sx; 0 sx cx;]

syms tetz tety tetx
z=[cos(tetz) -sin(tetz) 0; sin(tetz) cos(tetz) 0; 0 0 1]
y=[cos(tety) 0 +sin(tety); 0 1 0; -sin(tety) 0 cos(tety)]
x=[1 0 0; 0 cos(tetx) -sin(tetz); 0 sin(tetx) cos(tetx);]

syms  Chcx  Chcy  Chcz  Chsx  Chsy  Chsz
syms z y x

ChX=[1 0 0; 0 cos(x) +sin(x); 0 -sin(x) cos(x);]
ChY=[cos(y) 0 -sin(y); 0 1 0; +sin(y) 0 cos(y)]
ChZ=[cos(z) +sin(z) 0; -sin(z) cos(z) 0; 0 0 1]


% ChZ=[Chcz -Chsz 0; Chsz Chcz 0; 0 0 1]
% ChY=[Chcy 0 -Chsy; 0 1 0; Chsy 0 Chcy]
% ChX=[1 0 0; 0 Chcx -Chsx; 0 Chsx Chcx;]

ChronoXYZ=ChX*ChY*ChZ
Chrono=ChZ*ChX*ChY
%%% changing the rotation order from ZXY to XYZ  supported by chrono
clear
load('ROT.mat')
ROT=ans.ROT*pi/180;
NEWXYZ=zeros(size(ROT,1),3);
for i=1:size(ROT,1)
    x=ROT(i,1);
    y=ROT(i,2);
    z=ROT(i,3);
    Chy=asin(-cos(y)*sin(x)*sin(z) + cos(z)*sin(y));
    Chz=asin(cos(x)*sin(z)/cos(Chy));
    Chx=asin((sin(y)*sin(z) + cos(y)*cos(z)*sin(x))/cos(Chy));

   NEWXYZ(i,:)=[Chx Chy Chz]*180/pi;
end

