
# Risk Management – Expected Exposure, PFE & VaR


https://github.com/user-attachments/assets/001ee3d0-3d55-44b5-9919-995d20017a63


## Overview  
A unified framework for counterparty credit and market risk:  
- **Expected Exposure (EE)**  
- **Potential Future Exposure (PFE)**  
- **Value at Risk (VaR)** (Monte Carlo & Historical)  
- **Expected Shortfall (ES)**  
- **Netting effects**

## Installation  
```bash
git clone https://github.com/shubh123a3/Risk-Management-EE-PFE-VaR-HVaR.git
cd Risk-Management-EE-PFE-VaR-HVaR
pip install -r requirements.txt
```

## Usage  
```python
from main import mainCalculation, mainCode

# Monte Carlo EE/PFE & VaR/ES
mainCalculation()

# Historical VaR & ES via bootstrapped curves
mainCode()
```

## Theory & Mathematics

### 1. Positive Exposure  
$$
E(t) = \max\bigl(V(t),\,0\bigr)
$$

### 2. Expected Exposure (EE)  
$$
EE(t)
= \mathbb{E}^{\mathbb{Q}}\!\Bigl[\frac{E(t)}{M_t}\Bigr],
\quad
M_t = \exp\!\Bigl(\int_{0}^{t} r_s\,ds\Bigr)
$$

### 3. Potential Future Exposure (PFE)  
$$
PFE_\alpha(t)
= \inf\{\,x : \Pr(E(t)\le x)\ge \alpha\}
$$

### 4. Netting Effect  
$$
E_{\rm net}(t)
= \max\Bigl(\sum_{i=1}^n V_i(t),\,0\Bigr)
$$

### 5. Historical VaR  
Given P&L scenarios \(\Delta V_j = V^0 - V^{(j)}\):  
$$
VaR_\alpha
= \inf\{\,x : F_{\Delta V}(x)\ge \alpha\}
$$

### 6. Expected Shortfall (ES)  
$$
ES_\alpha
= \mathbb{E}\bigl[\Delta V \mid \Delta V \le VaR_\alpha\bigr]
= \frac{1}{\alpha}\int_{-\infty}^{VaR_\alpha} x\,dF_{\Delta V}(x)
$$

### 7. Monte Carlo VaR (Hull–White Model)  
Short‑rate SDE:  
$$
dr_t = \lambda\bigl(\theta(t)-r_t\bigr)\,dt + \eta\,dW_t
$$  
Euler discretization:  
$$
r_{t+\Delta t}
= r_t + \lambda\bigl(\theta(t)-r_t\bigr)\,\Delta t
+ \eta\,\sqrt{\Delta t}\,\xi_t,\quad \xi_t\sim N(0,1)
$$

## Output  
- Time‑series plots of EE vs PFE  
- Histograms of portfolio P&L with VaR & ES markers  
- Netting comparison charts  

## License  
MIT © Shubh Shrishrimal  
```
