import matplotlib.pyplot as plt

weight = [68, 81, 64, 56, 78, 74, 61, 77, 66, 68, 59, 71,
          80, 59, 67, 81, 69, 73, 69, 74, 70, 65]

plt.hist(weight, bins=30, label='bins=30', width=0.4, cumulative=True, histtype='barstacked')
plt.legend()
plt.show()