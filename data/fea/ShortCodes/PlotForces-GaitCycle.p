#!/usr/bin/gnuplot
set terminal png size 3072,1920 enhanced font "Helvetica,20"
set output "Forces.png"

set multiplot layout 3, 1 
set tmargin 3
set title "F_x-Anetrio" font ",30"
set tics font ",30"
set grid
set xrange [0.03:1.3]
plot 'femur.txt' using 1:2 with l lt 1 lw 6 lc rgb "black" title 'Femur',\
	 'tibia1.txt' using 1:(-1*$2) with l lt 1 lw 6 lc rgb "blue" title 'Lateral',\
	 'tibia2.txt' using 1:(-1*$2) with l lt 1 lw 6 lc rgb "red" title 'Medial'

unset key
set title "F_y-Superior"
set tics font ",30"
set grid
set xrange [0.03:1.3]
plot 'femur.txt' using 1:3 with l lt 1 lw 6 lc rgb "black" title 'Femur',\
	 'tibia1.txt' using 1:(-1*$3) with l lt 1 lw 6 lc rgb "blue" title 'Lateral',\
	 'tibia2.txt' using 1:(-1*$3) with l lt 1 lw 6 lc rgb "red" title 'Medial'

unset key
set title "F_z-Lateral"
set tics font ",30"
set label font ",30"
set xlabel 'Time(s)' font ",25"
set grid
set xrange [0.03:1.3]
plot 'femur.txt' using 1:4 with l lt 1 lw 6 lc rgb "black" title 'Femur',\
	 'tibia1.txt' using 1:(-1*$4) with l lt 1 lw 6 lc rgb "blue" title 'Lateral',\
	 'tibia2.txt' using 1:(-1*$4) with l lt 1 lw 6 lc rgb "red" title 'Medial'
unset key
unset multiplot
set terminal png
replot
set terminal x11
pause 1