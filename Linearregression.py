import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

#Setting up variables for the data frames
MU_frame = pd.read_csv('MU_history.csv')
NVDA_frame = pd.read_csv('NVDA_history.csv')
SPX_frame = pd.read_csv('SPX_history.csv')

#Checking out data tables heads
#print(MU_frame.head())
#print(NVDA_frame.head())
#print(SPX_frame.head())

#Percent change of Adj Close for each stock and index removing NaN values
MU_pct = MU_frame['Adj Close'].pct_change().dropna().values
SPX_pct = SPX_frame['Adj Close'].pct_change().dropna().values
NVDA_pct = NVDA_frame['Adj Close'].pct_change().dropna().values

#Regression of each stock on SPX
MU_fit = stats.linregress(SPX_pct, MU_pct)
NVDA_fit = stats.linregress(SPX_pct, NVDA_pct)
print("MU beta =", MU_fit.slope)
print("NVDA beta =", NVDA_fit.slope)

#Plot points
x = np.linspace(np.amin(SPX_pct), np.amax(SPX_pct))
plt.plot(SPX_pct, MU_pct, 'b.', alpha=0.4, label='MU')
plt.plot(SPX_pct, NVDA_pct, 'r.', alpha=0.4, label='NVDA')

#Plot lines
plt.plot(x, MU_fit.slope * x + MU_fit.intercept, color='blue', label=f'MU fit (R² = {MU_fit.rvalue**2:.2f})')
plt.plot(x, NVDA_fit.slope * x + NVDA_fit.intercept, color='red', label=f'NVDA fit (R² = {NVDA_fit.rvalue**2:.2f})')
plt.grid(True)
plt.xlabel('SPX daily return')
plt.ylabel('Stock daily return')
plt.legend()
plt.title(f'MU β = {MU_fit.slope:.2f}   |   NVDA β = {NVDA_fit.slope:.2f}')
plt.show()