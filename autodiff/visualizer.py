import matplotlib.pyplot as plt
from autodiff import *

x = var('x')
y = var('y')

f = x*y
print (f.eval(x=2, y=2))
#f.visualize(x=2, y=2)







fig  = plt.figure( figsize = (10,10))
fig.patch.set_facecolor('black')
plt.gca().axis('off')
circle = plt.Circle((0, 0), radius=0.2, fc='none', ec='gray', lw=10)
plt.gca().add_patch(circle)

plt.text(0, 0, 'X1', fontdict={'size':20}, c='white')

plt.axis('scaled')
plt.show()

