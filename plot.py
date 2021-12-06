import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

fig = plt.figure()
x = np.arange(0, 2*np.pi, 0.01) # x-array
line, = plt.plot(x, np.sin(x))
def animate(i):
    line.set_ydata(np.sin(x + i/10.0)) # update the data
    return line
ani.FuncAnimation(fig, animate, np.arange(1, 200), interval=25, blit=False)
plt.show()

# plt.savefig('/Users/liujinyi/a-byte-of-python-bnu-rst/pic/matplotlib_me/ball_2.png', dpi=300)
# plt.show()
