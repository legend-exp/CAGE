##########################################

#The following code is used to determine where to place a disk source for a given collimator position and rotation around the x-axis. This can be used to specify the source position in g4simple geometries and run files.

##########################################

import math
a = input('Input the x position of the collimator in mm: ')
b = input('The y position: ')
c = input('The z position: ')
d = input('Input the angle that the collimator is rotated in degrees: ')
e = input('Input the thickness of the collimator in mm: ')
f = input('Input the thickness of the source disk in mm: ')
g = float(a)
h = float(b)
i = float(c)
j = float(d)
k = float(e)
l = float(f)
m = math.radians(j)
n = float(m)
o = math.sin(n)
p = math.cos(n)
q = float(o)
r = float(p)
s = h+q*(k/2+l/2+0.1)
t = i+r*(k/2+l/2+0.1)
print('To be directly behind the collimator, the source should be placed at (x,y,z)=({},{},{})'.format(a,s,t))
