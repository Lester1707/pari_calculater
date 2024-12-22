import numpy as np
from scipy.integrate import cumtrapz
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def logit(x:float):
    return np.log(x/(1. - x))

def sigm(x:float) ->float:
    return 1./(1. + np.exp(-x))

def s_d(q:float, p:float, k:float) -> float:
    return (k*p - k*q + q - 1.) / (((1. - q)**p)*(k*q - q + 1.)**(1. - p))

def s(q:float, p:float, k:float) -> float:
    return (1. - q + k*q)**p*(1. - q)**(1. - p)

def p_norm(x:float, mu:float, sigma2:float) -> float:
    return 1. / np.sqrt(2.*np.pi*sigma2) * np.exp(-(x - mu)**2./(2.*sigma2))

# Определяем функцию
def f_dir(x:float, mu:float, sigma2:float, q:float, k:float) -> float:
    return p_norm(x, mu, sigma2)*s_d(q, sigm(x), k)

def f(x:float, mu:float, sigma2:float, q:float, k:float) -> float:
    return p_norm(x, mu, sigma2)*s(q, sigm(x), k)




Q = np.linspace(0.001, 0.9999, 5000)
p = 0.8
mu = logit(p)
sigma2 = 10
k = 1.8

y = [quad(lambda x: f_dir(x, mu, sigma2, q, k), -np.inf, np.inf)[0] for q in Q]
sol_number = np.abs(y).argmin()
eps = 0.01
print(f"default q: {(k*p - 1)/(k - 1)}")
print(f"solution {f'find at {Q[sol_number]}' if np.abs(y[sol_number]) < eps else 'not find'}")
q_opt_mod = Q[sol_number]

x = Q

q_opt = (k*p - 1)/(k - 1)
print(q_opt)
# Plot cumulative returns
plt.figure(figsize=(10, 4))  # Adjust aspect ratio
plt.plot(x, s(x, p-0.12, k), label=f'S(q, p={0.68}, k={k})', color='purple')
#plt.plot(x, y, label=f'S_mod(q, p={p}, k={k})', color='orange')
plt.axhline(y=0, color="black", linewidth=0.8)  # Horizontal at y=0
plt.axvline(x=0, color="black", linewidth=0.8)  # Vertical at x=0 (useful
plt.axhline(y=1, color="green", linewidth=1, linestyle="--", xmin=0)  # Horizontal reference line
plt.ylim((plt.ylim()[0], plt.ylim()[1]+0.03))
if q_opt > 0:
    plt.axvline(q_opt, linestyle="--", color="blue")
    plt.text(
    x=q_opt+0.02,  # Middle of the x-axis
    y=(plt.ylim()[0] + plt.ylim()[1]) / 2,                  # Slightly above y=0
    s="q_opt_base",       # Text to display
    color="blue",
    fontsize=9,
    ha="center",             # Horizontal alignment
    va="bottom",              # Vertical alignment
    rotation='vertical'
    )
if q_opt_mod:
    plt.axvline(q_opt_mod, linestyle="--", color='red')
    plt.text(
    x=q_opt_mod+0.02,  # Middle of the x-axis
    y=(plt.ylim()[0] + plt.ylim()[1]) / 2,                  # Slightly above y=0
    s="q_opt_modified",       # Text to display
    color="red",
    fontsize=9,
    ha="center",             # Horizontal alignment
    va="bottom",              # Vertical alignment
    rotation='vertical'
    )

plt.text(
    x=0.4,  # Middle of the x-axis
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
