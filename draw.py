import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def s_d(q:float, p:float, k:float) -> float:
    return (k*p - k*q + q - 1.) / (((1. - q)**p)*(k*q - q + 1.)**(1. - p))

def s(q:float, p:float, k:float) -> float:
    return (1 - q + k*q)**p*(1 - q)**(1 - p)

x = np.linspace(0, 0.999999, num=1000)
k = 1.5
p = 0.5

q_opt = (k*p - 1)/(k - 1)
print(q_opt)
# Plot cumulative returns
plt.figure(figsize=(10, 4))  # Adjust aspect ratio
plt.plot(x, s(x, p-0.2, k), label=f'S(q, p={p}, k={k})', color='purple')
plt.axhline(y=0, color="black", linewidth=0.8)  # Horizontal at y=0
plt.axvline(x=0, color="black", linewidth=0.8)  # Vertical at x=0 (useful
plt.axhline(y=1, color="green", linewidth=1, linestyle="--", xmin=0)  # Horizontal reference line
plt.ylim((plt.ylim()[0], plt.ylim()[1]+0.03))
if q_opt > 0:
    plt.axvline(q_opt, linestyle="--", color='red')
    plt.text(
    x=q_opt+0.02,  # Middle of the x-axis
    y=(plt.ylim()[0] + plt.ylim()[1]) / 2,                  # Slightly above y=0
    s="q_opt",       # Text to display
    color="red",
    fontsize=9,
    ha="center",             # Horizontal alignment
    va="bottom",              # Vertical alignment
    rotation='vertical'
    )

plt.text(
    x=0.45,  # Middle of the x-axis
    y=1 + 0.01,                  # Slightly above y=0
    s="The benefit line",       # Text to display
    color="green",
    fontsize=9,
    ha="center",             # Horizontal alignment
    va="bottom",              # Vertical alignment
)

# plt.annotate(
#     "The area of sharp bank failure",                      # Text
#     xy=(0.99 , s(0.99, p, k)),  # Point to annotate
#     xytext=(0.99-0.3, s(0.99, p, k) - 0.2),  # Position of text
#     arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),  # Arrow style
#     fontsize=10,
#     color="black",
# )

# Customize the style to match the image
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Horizontal reference line
plt.xlabel("q")
plt.ylabel("S value", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend(fontsize=8, loc="best", frameon=False)  # Match legend style
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Fewer x-ticks



# Rotate x-ticks for better visibility
plt.xticks(rotation=30, ha="right")

# Tight layout for spacing
plt.tight_layout()

# Display plot
plt.show()