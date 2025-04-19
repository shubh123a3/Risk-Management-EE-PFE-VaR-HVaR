
# Risk Management – Expected Exposure, Potential Future Exposure & VaR

A comprehensive toolkit for credit & market risk analysis:  
- **Expected Exposure (EE)**  
- **Potential Future Exposure (PFE)**  
- **Value at Risk (VaR)** via Monte Carlo & Historical scenarios  
- **Expected Shortfall (ES)**  
- **Netting effects**

---

## 📖 Overview  
This project simulates interest‑rate and portfolio P&L distributions to compute counterparty credit metrics (EE/PFE) and market‑risk measures (VaR/ES). It supports:  
- Monte Carlo under the Hull–White model  
- Historical shock scenarios  
- Netting agreements

---

## ⚙️ Installation  
```bash
git clone https://github.com/shubh123a3/Risk-Management-EE-PFE-VaR-HVaR.git
cd Risk-Management-EE-PFE-VaR-HVaR
pip install -r requirements.txt
```

---

## 🚀 Usage  
```python
from main import mainCalculation, mainCode
# Monte Carlo EE/PFE/Monte Carlo VaR/ES
mainCalculation()
# Historical VaR/ES via bootstrapped yield curves
mainCode()
```

---

## 🔬 Theory & Mathematics

### 1. Positive Exposure  
$$
E(t) \;=\; \max\bigl(V(t),\,0\bigr)
$$

### 2. Expected Exposure (EE)  
Discounted average positive exposure under \(\mathbb{Q}\):  
$$
\mathrm{EE}(t)
=\mathbb{E}^{\mathbb{Q}}\!\Bigl[\tfrac{E(t)}{M_t}\Bigr],\quad
M_t=\exp\!\Bigl(\!\int_{0}^{t}r_s\,ds\Bigr)
$$

### 3. Potential Future Exposure (PFE)  
\(\alpha\)-quantile of exposure:  
$$
\mathrm{PFE}_\alpha(t)
=\inf\{x:\Pr(E(t)\le x)\ge\alpha\}
$$

### 4. Netting Effect  
Combining \(n\) trades reduces gross exposure:  
$$
E_{\rm net}(t)
=\max\Bigl(\sum_{i=1}^n V_i(t),0\Bigr)
$$

### 5. Value at Risk (VaR)  
- **Historical VaR**:  
  P&L scenarios \(\Delta V_j = V^0 - V^{(j)}\), then  
  $$
  \mathrm{VaR}_\alpha
  =\inf\{x: F_{\Delta V}(x)\ge\alpha\}
  $$
- **Monte Carlo VaR**: simulate \(N\) paths under Hull–White, compute \(\Delta V_i\), then quantile.

### 6. Expected Shortfall (ES)  
Average loss beyond VaR:  
$$
\mathrm{ES}_\alpha
=\mathbb{E}\bigl[\Delta V \mid \Delta V\ge \mathrm{VaR}_\alpha\bigr]
=\frac{1}{1-\alpha}\int_{\mathrm{VaR}_\alpha}^{\infty} x\,dF(x)
$$

### 7. Hull–White Model (Monte Carlo)  
Short‑rate SDE:  
$$
dr_t=\lambda\bigl(\theta(t)-r_t\bigr)\,dt+\eta\,dW_t
$$  
Euler discretization:  
$$
r_{t+\Delta t}=r_t+\lambda(\theta(t)-r_t)\,\Delta t+\eta\,\sqrt{\Delta t}\,\xi_t,\;\xi_t\sim N(0,1)
$$  

---

## 📊 Output  
- Time‑series plots of EE vs PFE  
- Histograms of portfolio P&L with VaR & ES markers  
- Netting comparison charts  

---

## 🏷️ License  
MIT © Shubh Shrishrimal  
```
