import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y =[0.2517539816664106, 0.8709278350515464, 0.908959313446725, 0.9256309474555235]


plt.plot(x, y, color = 'red', linestyle = 'dashed', linewidth = 2, marker ='o', markersize=6,
         markerfacecolor = 'red', markeredgecolor='red')

plt.grid()
plt.show()
