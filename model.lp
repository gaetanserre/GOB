\* Source Pyomo model name=unknown *\

min 
x1:
+1 x2
+1 x3
+1 x4
+1 x5
+1 x6

s.t.

c_e_x7_:
+1 x8
+1 x9
= 1

c_u_x10_:
+2.0 x8
+2.0 x9
-1 x2
<= 0

c_u_x11_:
+2.0 x8
+2.0 x9
-1 x3
<= 0

c_u_x12_:
+16.0 x8
+24.0 x9
-1 x4
<= 0

c_u_x13_:
+18.0 x8
+26.0 x9
-1 x5
<= 0

c_u_x14_:
+20.0 x8
+28.0 x9
-1 x6
<= 0

bounds
   0 <= x8 <= +inf
   0 <= x9 <= +inf
   0 <= x2 <= +inf
   0 <= x3 <= +inf
   0 <= x4 <= +inf
   0 <= x5 <= +inf
   0 <= x6 <= +inf
end
