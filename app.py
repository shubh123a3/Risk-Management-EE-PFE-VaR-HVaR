import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import enum

import Monte_Var
import HVar

class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)

    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) + eta * eta / (
                2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))

    # theta = lambda t: 0.1 +t -t
    # print("changed theta")

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    R[:, 0] = r0
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt + eta * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    # Outputs
    paths = {"time": time, "R": R}
    return paths


def HW_theta(lambd, eta, P0T):
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) + eta * eta / (
                2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    # print("CHANGED THETA")
    return theta  # lambda t: 0.1+t-t


def HW_A(lambd, eta, P0T, T1, T2):
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau: 1.0 / lambd * (np.exp(-lambd * tau) - 1.0)
    theta = HW_theta(lambd, eta, P0T)
    temp1 = lambd * np.trapz(theta(T2 - zGrid) * B_r(zGrid), zGrid)

    temp2 = eta * eta / (4.0 * np.power(lambd, 3.0)) * (
                np.exp(-2.0 * lambd * tau) * (4 * np.exp(lambd * tau) - 1.0) - 3.0) + eta * eta * tau / (
                        2.0 * lambd * lambd)

    return temp1 + temp2


def HW_B(lambd, eta, T1, T2):
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)


def HW_ZCB(lambd, eta, P0T, T1, T2, rT1):
    n = np.size(rT1)

    if T1 < T2:
        B_r = HW_B(lambd, eta, T1, T2)
        A_r = HW_A(lambd, eta, P0T, T1, T2)
        return np.exp(A_r + B_r * rT1)
    else:
        return np.ones([n])


def HWMean_r(P0T, lambd, eta, T):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2.0 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 2500)
    temp = lambda z: theta(z) * np.exp(-lambd * (T - z))
    r_mean = r0 * np.exp(-lambd * T) + lambd * np.trapz(temp(zGrid), zGrid)
    return r_mean


def HW_r_0(P0T, lambd, eta):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0


def HW_Mu_FrwdMeasure(P0T, lambd, eta, T):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 500)

    theta_hat = lambda t, T: theta(t) + eta * eta / lambd * 1.0 / lambd * (np.exp(-lambd * (T - t)) - 1.0)

    temp = lambda z: theta_hat(z, T) * np.exp(-lambd * (T - z))

    r_mean = r0 * np.exp(-lambd * T) + lambd * np.trapz(temp(zGrid), zGrid)

    return r_mean


def HWVar_r(lambd, eta, T):
    return eta * eta / (2.0 * lambd) * (1.0 - np.exp(-2.0 * lambd * T))


def HWDensity(P0T, lambd, eta, T):
    r_mean = HWMean_r(P0T, lambd, eta, T)
    r_var = HWVar_r(lambd, eta, T)
    return lambda x: stats.norm.pdf(x, r_mean, np.sqrt(r_var))


def HW_SwapPrice(CP, notional, K, t, Ti, Tm, n, r_t, P0T, lambd, eta):
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t

    global swap
    if n == 1:
        ti_grid = np.array([Ti, Tm])
    else:
        ti_grid = np.linspace(Ti, Tm, n)
    tau = ti_grid[1] - ti_grid[0]

    # overwrite Ti if t>Ti
    prevTi = ti_grid[np.where(ti_grid < t)]
    if np.size(prevTi) > 0:  # prevTi != []:
        Ti = prevTi[-1]

    # Now we need to handle the case when some payments are already done
    ti_grid = ti_grid[np.where(ti_grid > t)]

    temp = np.zeros(np.size(r_t));

    P_t_TiLambda = lambda Ti: HW_ZCB(lambd, eta, P0T, t, Ti, r_t)

    for (idx, ti) in enumerate(ti_grid):
        if ti > Ti:
            temp = temp + tau * P_t_TiLambda(ti)

    P_t_Ti = P_t_TiLambda(Ti)
    P_t_Tm = P_t_TiLambda(Tm)

    if CP == OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP == OptionTypeSwap.RECEIVER:
        swap = K * temp - (P_t_Ti - P_t_Tm)

    return swap * notional
st.title('Risk Management -Expected Exposure ,Potential Future Exposure and Value at Risk (VaR) using Monte Carlo Simulation, Historical Var,Expected Shortfall')
st.write('This app calculates Expected Exposure, Potential Future Exposure and Value at Risk (VaR) using Monte Carlo Simulation, Historical Var,Expected Shortfall for a given set of parameters.')

Risk_method =st.radio('Select the Risk Measure :', ('Expected Exposure ','Monte Carlo Simulation Var', 'Historical VaR'))


if Risk_method=='Expected Exposure ':
    st.write('Expected Exposure is a measure of the average expected future exposure of a portfolio of financial instruments. It is calculated as the average of the potential future exposure over a specified time horizon.')
    st.write('The formula for Expected Exposure is:')
    st.latex(r'''
        EE = \frac{1}{N} \sum_{i=1}^{N} PFE_i
    ''')
    st.write('Where:')
    st.write('EE = Expected Exposure')
    st.write('PFE = Potential Future Exposure')
    st.write('N = Number of simulations')

    NoOfPaths =2000
    NoOfSteps = 1000
    lambd = 0.5
    eta = 0.03
    notional = st.sidebar.slider('Notional:', min_value=1000.0, max_value=1000000.0, value=10000.0)
    notional2 = st.sidebar.slider('Notional 2:', min_value=1000.0, max_value=1000000.0, value=10000.0)
    alpha = 0.99
    alpha2 = 0.95
    st.subheader('confidence interval is of 95% and 99%')
    P0T = lambda T: np.exp(-0.1 * T)
    r0 = HW_r_0(P0T, lambd, eta)
    # In this experiment we compare ZCB from the Market and Analytical expression
    N = 25
    T_end =50
    Tgrid = np.linspace(0, T_end, N)

    Exact = np.zeros([N, 1])
    Proxy = np.zeros([N, 1])
    for i, Ti in enumerate(Tgrid):
        Proxy[i] = HW_ZCB(lambd, eta, P0T, 0.0, Ti, r0)
        Exact[i] = P0T(Ti)
    fig1=plt.figure(1)
    plt.grid()
    plt.plot(Tgrid, Exact, '-k')
    plt.plot(Tgrid, Proxy, '--r')
    plt.legend(["Analytcal ZCB", "Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')
    st.pyplot(fig1)
    # Swap settings
    K = st.sidebar.slider('Strike:', min_value=0.0, max_value=1.0, value=0.01)
    Ti = st.sidebar.slider('Begining of the swap:', min_value=0.0, max_value=20.0, value=1.0)
    Tm = st.sidebar.slider('End date of the swap:', min_value=0.0, max_value=20.0, value=10.0)
    n = st.sidebar.slider('Number of payments between Ti and Tm:', min_value=1, max_value=100, value=10)

    paths = GeneratePathsHWEuler(NoOfPaths, NoOfSteps, Tm + 1.0, P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1] - timeGrid[0]
    M_t = np.zeros([NoOfPaths, NoOfSteps])

    for i in range(0, NoOfPaths):
        M_t[i, :] = np.exp(np.cumsum(r[i, 0:-1]) * dt)

        # protfolio without netung
    Value = np.zeros([NoOfPaths, NoOfSteps + 1])
    E = np.zeros([NoOfPaths, NoOfSteps + 1])
    EE = np.zeros([NoOfSteps + 1])
    PFE = np.zeros([NoOfSteps + 1])
    PFE2 = np.zeros([NoOfSteps + 1])
    for idx, ti in enumerate(timeGrid[0:-2]):
        V = HW_SwapPrice(OptionTypeSwap.PAYER, notional, K, timeGrid[idx], Ti, Tm, n, r[:, idx], P0T, lambd, eta)
        Value[:, idx] = V
        E[:, idx] = np.maximum(V, 0.0)
        EE[idx] = np.mean(E[:, idx] / M_t[:, idx])
        PFE[idx] = np.quantile(E[:, idx], alpha)
        PFE2[idx] = np.quantile(E[:, idx], alpha2)

    # portfolio with nettinh
    ValuePort = np.zeros([NoOfPaths, NoOfSteps + 1])
    EPort = np.zeros([NoOfPaths, NoOfSteps + 1])
    EEPort = np.zeros([NoOfSteps + 1])
    PFEPort = np.zeros([NoOfSteps + 1])

    for (idx, ti) in enumerate(timeGrid[0:-2]):
        Swap1 = HW_SwapPrice(OptionTypeSwap.PAYER, notional, K, timeGrid[idx], Ti, Tm, n, r[:, idx], P0T, lambd, eta)
        Swap2 = HW_SwapPrice(OptionTypeSwap.RECEIVER, notional2, 0.0, timeGrid[idx], Tm - 2.0 * (Tm - Ti) / n, Tm, 1,
                             r[:, idx], P0T, lambd, eta)

        VPort = Swap1 + Swap2
        ValuePort[:, idx] = VPort
        EPort[:, idx] = np.maximum(VPort, 0.0)
        EEPort[idx] = np.mean(EPort[:, idx] / M_t[:, idx])
        PFEPort[idx] = np.quantile(EPort[:, idx], alpha)
    # Plotting
    co1, co2 = st.columns(2)
    with co1:

        fig2=plt.figure(2)
        plt.plot(timeGrid, Value[0:100, :].transpose(), 'b')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('exposure, Value(t)')
        plt.title('Value of a swap')
        st.pyplot(fig2)
    with co2:


        fig3=plt.figure(3)
        plt.plot(timeGrid, E[0:100, :].T, 'r')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('exposure, E(t)')
        plt.title('Positive Exposure E(t)')
        st.pyplot(fig3)

    col3, col4 = st.columns(2)
    with col3:

        fig4=plt.figure(4)
        plt.plot(timeGrid, EE, 'r')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('exposure, EE(t)')
        plt.title('Discounted Expected (positive) exposure, EE')
        plt.legend(['EE', 'PFE'])
        st.pyplot(fig4)
    with col4:

            fig5=plt.figure(5)
            plt.plot(timeGrid, EE, 'r')
            plt.plot(timeGrid, PFE, 'k')
            plt.plot(timeGrid, PFE2, '--b')
            plt.grid()
            plt.xlabel('time')
            plt.ylabel(['EE, PEE(t)'])
            plt.title('Discounted Expected (positive) exposure, EE')
            plt.legend(['EE', 'PFE', 'PFE2'])
            st.pyplot(fig5)
    # Portfolio with two swaps
    col5, col6 = st.columns(2)
    with col5:
        st.write('Portfolio consits of  two swaps')
        fig6=plt.figure(6)
        plt.plot(timeGrid, EEPort, 'r')
        plt.plot(timeGrid, PFEPort, 'k')
        plt.grid()
        plt.title('Portfolio with two swaps')
        plt.legend(['EE-port', 'PFE-port'])
        st.pyplot(fig6)



    with col6:
        st.write('Comparison of EEs and PFEs ')
        st.write('EE is the expected exposure and PFE is the potential future exposure')
        st.write('The expected exposure is the average of the potential future exposure over a specified time horizon.')
        fig7=plt.figure(7)
        plt.plot(timeGrid, EE, 'r')
        plt.plot(timeGrid, EEPort, '--r')
        plt.grid()
        plt.title('Comparison of EEs ')
        plt.legend(['EE, swap', 'EE, portfolio'])
        st.pyplot(fig7)

    st.write(' between netting and non-netting ')

    fig8=plt.figure(8)
    plt.plot(timeGrid, PFE, 'k')
    plt.plot(timeGrid, PFEPort, '--k')
    plt.grid()
    plt.title('Comparison of PFEs ')
    plt.legend(['PFE, swap', 'PFE, portfolio'])
    st.pyplot(fig8)

if Risk_method=='Monte Carlo Simulation Var':
    st.write('Monte Carlo Simulation VaR is a method used to estimate the Value at Risk (VaR) of a portfolio of financial instruments using Monte Carlo simulation. It involves generating a large number of random scenarios for the future value of the portfolio and calculating the VaR based on these scenarios.')
    st.write('The formula for Monte Carlo Simulation VaR is:')
    st.latex(r'''
        VaR = -\frac{1}{N} \sum_{i=1}^{N} V_i
    ''')
    st.write('Where:')
    st.write('VaR = Value at Risk')
    st.write('V = Portfolio value')
    st.write('N = Number of simulations')

    NoOfPaths =st.sidebar.slider('Number of Paths:', min_value=100, max_value=10000, value=2000)
    NoOfSteps = st.sidebar.slider('Number of Steps:', min_value=10, max_value=10000, value=100)
    lambd = st.sidebar.slider('Lambda:', min_value=0.0, max_value=1.0, value=0.5)
    eta = st.sidebar.slider('Eta:', min_value=0.0, max_value=1.0, value=0.03)
    notional = st.sidebar.slider('Notional:', min_value=1000.0, max_value=1000000.0, value=10000.0)
    alpha = st.sidebar.slider('Alpha: Confidence interval(1.0-alpha) ', min_value=0.01, max_value=1.0, value=0.05)
    st.subheader('confidence interval is of ' + str(1.0-alpha) + ' %')





    # We define a ZCB curve (obtained from the market)
    P0T = lambda T: np.exp(-0.001 * T)
    r0 = Monte_Var.HW_r_0(P0T, lambd, eta)

    # In this experiment we compare ZCB from the Market and Analytical expression
    N = 25
    T_end = 50
    Tgrid = np.linspace(0, T_end, N)

    Exact = np.zeros([N, 1])
    Proxy = np.zeros([N, 1])
    for i, Ti in enumerate(Tgrid):
        Proxy[i] = Monte_Var.HW_ZCB(lambd, eta, P0T, 0.0, Ti, r0)
        Exact[i] = P0T(Ti)

    fig1=plt.figure(1)
    plt.grid()
    plt.plot(Tgrid, Exact, '-k')
    plt.plot(Tgrid, Proxy, '--r')
    plt.legend(["Analytcal ZCB", "Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')
    st.pyplot(fig1)

    T_end = st.sidebar.slider('End date of the swap:', min_value=0.0, max_value=50.0, value=20.0)
    paths = Monte_Var.GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T_end, P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt = timeGrid[1] - timeGrid[0]

    M_t = np.zeros([NoOfPaths, NoOfSteps])

    for i in range(0, NoOfPaths):
        M_t[i, :] = np.exp(np.cumsum(r[i, 0:-1]) * dt)

    r0 = r[0, 0]
    stepSize = 10
    V_M = np.zeros([NoOfPaths, NoOfSteps - stepSize])

    for i in range(0, NoOfSteps - stepSize):
        dr = r[:, i + stepSize] - r[:, i]
        V_t0 = Monte_Var.Portfolio(P0T, r[:, 0] + dr, lambd, eta)
        V_M[:, i] = V_t0
    fig2, ax = plt.subplots(figsize=(12, 6))
    V_t0_vec = V_M.flatten()

    # Histogram
    ax.hist(V_t0_vec, bins=100, alpha=0.75, edgecolor='black')

    st.subheader('Value of Portfolio V(t_0)= '+ str( Monte_Var.Portfolio(P0T, r[0, 0], lambd, eta)))
    st.write('confidence interval of ' + str(1.0-alpha) + ' %')

    HVaR_estimate = np.quantile(V_t0_vec, alpha)
    st.subheader('(H)VaR for alpha = ' +  str(alpha) + ' is equal to=' + str( HVaR_estimate))

    # Expected shortfal
    st.header('Expected shortfall')
    condLosses = V_t0_vec[V_t0_vec < HVaR_estimate]
    st.write('P&L which < VaR_alpha =', condLosses)
    ES = np.mean(condLosses)
    st.subheader('Expected shortfal = ' + str( ES))

    ax.axvline(HVaR_estimate, color='red', linestyle='--', linewidth=3,
               label=f'VaR ({(1 - alpha) * 100:.1f}% α) = {HVaR_estimate:.2f}')
    ax.axvline(ES, color='black', linestyle='-.', linewidth=3,
               label=f'ES = {ES:.2f}')

    # Labels and styling
    ax.set_title('Portfolio P&L Distribution with VaR & ES', fontsize=18)
    ax.set_xlabel('Portfolio P&L', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend(fontsize=12, loc='upper right')
    st.pyplot(fig2)

    st.write('The histogram shows the distribution of portfolio P&L with VaR and ES lines.')

if Risk_method=='Historical VaR':
    st.write('Historical VaR is a method used to estimate the Value at Risk (VaR) of a portfolio of financial instruments based on historical data. It involves calculating the VaR based on the historical returns of the portfolio.')
    st.write('The formula for Historical VaR is:')
    st.latex(r'''
        VaR = -\frac{1}{N} \sum_{i=1}^{N} V_i
    ''')
    st.write('Where:')
    st.write('VaR = Value at Risk')
    st.write('V = Portfolio value')
    st.write('N = Number of simulations')
    st.write('The historical VaR is calculated using the historical returns of the portfolio.')

    st.subheader("Portfolio consists of 8 swaps with different maturities and notional amounts " )

    marketdataXLS = pd.read_excel('MrktData.xlsx')
    # Divedie the qute by 100 for value
    alpha= st.sidebar.slider('Alpha: Confidence interval(1.0-alpha) ', min_value=0.01, max_value=1.0, value=0.05)

    marketData = np.array(marketdataXLS) / 100.0
    shape = np.shape(marketData)
    NoOfScan = shape[0]
    NoOfInsts = shape[1]

    Scenarios = np.zeros([NoOfScan - 1, NoOfInsts])
    for i in range(0, NoOfScan - 1):
        for j in range(0, NoOfInsts):
            Scenarios[i, j] = marketData[i + 1, j] - marketData[i, j]

        # Construct instruments for TODAY's curve
    Swaps_mrkt = np.array([0.08, 0.2, 0.4, 0.77, 1.07, 1.29, 1.82, 1.9]) / 100
    mat = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    # Given market quotes for swaps and scenarios we generate now "shocked" yield curves
    Swaps_markt_shocked = np.zeros([NoOfScan - 1, NoOfInsts])
    for i in range(0, NoOfScan - 1):
        for j in range(0, NoOfInsts):
            Swaps_markt_shocked[i, j] = Swaps_mrkt[j] + Scenarios[i, j]
    place_holder=st.empty()
    # For every shocked market scenario we build a yield curve
    YC_for_VaR = []
    for i in range(0, NoOfScan - 1):
        P0T, instruments = HVar.BuildYieldCurve(Swaps_markt_shocked[i, :], mat)
        YC_for_VaR.append(P0T)
        st.write('Scenario number', i, ' out of  ', NoOfScan - 1)

    # For every shocked yield curve we re-value the portfolio of interest rate derivatives
    PortfolioPV = np.zeros([NoOfScan - 1])
    for i in range(0, NoOfScan - 1):
        PortfolioPV[i] = HVar.Portfolio(YC_for_VaR[i])

    # current Yeild curve
    Yc_today, insts = HVar.BuildYieldCurve(Swaps_mrkt, mat)
    st.subheader('Current Portfolio PV is ' + str( HVar.Portfolio(Yc_today)))

    # Histograms and Var Calculatiosn


    fig1, ax = plt.subplots(figsize=(12, 6))


    # Histogram
    ax.hist(PortfolioPV, bins=100, alpha=0.75, edgecolor='black')

    # VaR calculation
    st.header('VaR calculation')

    st.write('confidence interval of ' + str(1.0-alpha) + ' %')
    HVaR_estimate = np.quantile(PortfolioPV, alpha)
    st.subheader('(H)VaR  is equal to= ' ,str( HVaR_estimate))

    # Expected shortfal
    condLosses = PortfolioPV[PortfolioPV < HVaR_estimate]
    st.write('P&L which < VaR_alpha =', condLosses)
    ES = np.mean(condLosses)
    st.header('Expected shortfall')
    st.subheader('Expected shortfal = ' + str( ES))

    ax.axvline(HVaR_estimate, color='red', linestyle='--', linewidth=3,
               label=f'VaR ({(1 - alpha) * 100:.1f}% α) = {HVaR_estimate:.2f}')
    ax.axvline(ES, color='black', linestyle='-.', linewidth=3,
               label=f'ES = {ES:.2f}')

    # Labels and styling
    ax.set_title('Portfolio P&L Distribution with VaR & ES', fontsize=18)
    ax.set_xlabel('Portfolio P&L', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend(fontsize=12, loc='upper right')
    place_holder.pyplot(fig1)



