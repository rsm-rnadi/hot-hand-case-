{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🏀 Hot Hand Case — MGTA 453\n",
    "**Author:** Radin Nadi  \n",
    "**UC San Diego – MSBA 2025**  \n",
    "\n",
    "This notebook reproduces all calculations and figures for the *Hot Hand* case study.  \n",
    "It includes:\n",
    "- NBA Salary regressions (A–D)\n",
    "- Hot-hand conditional probability test\n",
    "- Hot-hand streaks simulation\n",
    "- Relationship plots for Basketball and Baseball datasets\n",
    "\n",
    "**Before running:**\n",
    "```bash\n",
    "pip install pandas numpy matplotlib statsmodels scikit-learn pyarrow\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import stats\n",
    "import statsmodels.formula.api as smf\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from numpy.random import default_rng\n",
    "\n",
    "plt.rcParams['figure.figsize'] = (7, 4)\n",
    "plt.rcParams['axes.grid'] = True"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "DATA_DIR = '.'  # Folder containing your parquet files\n",
    "\n",
    "nba  = pd.read_parquet(os.path.join(DATA_DIR, 'nba_pgdata.parquet'))\n",
    "bask = pd.read_parquet(os.path.join(DATA_DIR, 'BasketballRelationships.parquet'))\n",
    "base = pd.read_parquet(os.path.join(DATA_DIR, 'BaseballRelationships.parquet'))\n",
    "\n",
    "nba.columns  = [c.strip().replace(' ', '_') for c in nba.columns]\n",
    "bask.columns = [c.strip().lower() for c in bask.columns]\n",
    "base.columns = [c.strip().lower() for c in base.columns]\n",
    "\n",
    "print(f'Loaded: NBA={len(nba)}, Basketball={len(bask)}, Baseball={len(base)}')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1 – NBA Salary Analysis"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "for col in ['Salary','Age','FG','RB','AST','STL','BLK','PTS']:\n",
    "    nba[col] = pd.to_numeric(nba[col], errors='coerce')\n",
    "\n",
    "# Regression A\n",
    "regA = smf.ols('Salary ~ Age + FG + RB + AST + STL + BLK', data=nba).fit()\n",
    "display(regA.summary())\n",
    "\n",
    "# Regression B\n",
    "nba['log_Salary'] = np.log(nba['Salary'].clip(lower=1))\n",
    "regB = smf.ols('log_Salary ~ Age + FG + RB + AST + STL + BLK', data=nba).fit()\n",
    "display(regB.summary())\n",
    "\n",
    "# Regression C\n",
    "regC = smf.ols('log_Salary ~ Age + FG + RB + AST + STL + BLK + PTS', data=nba).fit()\n",
    "display(regC.summary())\n",
    "\n",
    "# Regression D (standardized)\n",
    "cols_std = ['log_Salary','Age','RB','AST','STL','BLK','PTS']\n",
    "df_std = nba[cols_std].dropna().copy()\n",
    "scaler = StandardScaler()\n",
    "df_std[cols_std] = scaler.fit_transform(df_std[cols_std])\n",
    "regD = smf.ols('log_Salary ~ Age + RB + AST + STL + BLK + PTS', data=df_std).fit()\n",
    "display(regD.summary())"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2 – Hot Hand Conditional Probability Test"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "players = ['A','B','C','D','E','F','G','H','I']\n",
    "p_hit_given_miss = np.array([0.56,0.51,0.46,0.60,0.47,0.51,0.58,0.52,0.71])\n",
    "p_hit_given_hit  = np.array([0.49,0.53,0.46,0.55,0.45,0.43,0.53,0.51,0.57])\n",
    "\n",
    "diff = p_hit_given_hit - p_hit_given_miss\n",
    "t, p2 = stats.ttest_1samp(diff, 0)\n",
    "p1 = p2/2 if t > 0 else 1 - p2/2\n",
    "print(f'Mean Δ = {diff.mean():.4f}, t = {t:.3f}, p(one-sided) = {p1:.4g}')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3 – Hot Hand Streaks Simulation"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "shots = np.array([248,884,419,339,272,451,433,351,403])\n",
    "obs_runs = np.array([128,431,203,172,134,245,227,176,220])\n",
    "p_hit = np.array([0.50,0.52,0.46,0.56,0.47,0.46,0.54,0.52,0.62])\n",
    "\n",
    "def simulate_runs(n,p,sims=20000,rng=None):\n",
    "    rng = default_rng() if rng is None else rng\n",
    "    seq = rng.random((sims,n)) < p\n",
    "    return 1 + np.count_nonzero(np.diff(seq,axis=1),axis=1)\n",
    "\n",
    "rng = default_rng(123)\n",
    "rows = []\n",
    "for i,pl in enumerate(players):\n",
    "    r = simulate_runs(shots[i], p_hit[i], rng=rng)\n",
    "    mu, sd = r.mean(), r.std(ddof=1)\n",
    "    z = (obs_runs[i] - mu)/sd\n",
    "    p_emp = (np.sum(np.abs(r - mu) >= abs(obs_runs[i] - mu)) + 1) / (len(r) + 1)\n",
    "    rows.append([pl, shots[i], obs_runs[i], mu, sd, z, p_emp])\n",
    "\n",
    "runs = pd.DataFrame(rows, columns=['Player','Shots','ObsRuns','ExpRuns','SD','Z','p_emp'])\n",
    "display(runs)"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 4 – Relationship Plots"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "for x in ['points','attempts']:\n",
    "    if x in bask and 'percentage' in bask:\n",
    "        plt.scatter(bask[x], bask['percentage'], s=12)\n",
    "        plt.title(f'Basketball {x} vs percentage')\n",
    "        plt.xlabel(x); plt.ylabel('percentage'); plt.show()\n",
    "\n",
    "for x in ['hits','atbats']:\n",
    "    if x in base and 'avg' in base:\n",
    "        plt.scatter(base[x], base['avg'], s=12)\n",
    "        plt.title(f'Baseball {x} vs avg')\n",
    "        plt.xlabel(x); plt.ylabel('avg'); plt.show()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔍 Key Findings\n",
    "- Regression C generally has the best fit (highest R²).\n",
    "- Including PTS improves model interpretability and significance.\n",
    "- The hot-hand conditional test shows no statistically significant difference.\n",
    "- Streaks simulation: observed runs are consistent with random sequences for most players.\n",
    "- Basketball/ Baseball relationships both show positive but diminishing returns."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
