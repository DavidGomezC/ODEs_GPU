# import jax
# import jax.numpy as jnp
# import diffrax
# import equinox as eqx
# import lineax as lx
# import optimistix as optx
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import random
import time
import os 
from tkinter import simpledialog
from operator import contains
from tkinter.filedialog import askdirectory
import tkinter as tk
from ipywidgets import interact, widgets, fixed, interactive, HBox, Layout, VBox
import pickle
import matplotlib as mpl
import pandas as pd


class Repressilator:
    def __init__(self, tspan: tuple, tpts: int, theta: np.array, y0:np.array):

        # # Check that tspan is a tuple
        # if not isinstance(tspan, tuple):
        #     raise ValueError(f"Expected a tuple for tspan, got {type(tspan)}")
        
        # # Check that tpts is an integer
        # if not isinstance(tpts, int):
        #     raise ValueError(f"Expected an integer for tpts, got {type(tpts)}")
        
        # # Check that theta and y0 are 2D arrays, otherwise simulation won't work
        # if len(np.shape(theta)) != 2 or len(np.shape(y0)) != 2:
        #     raise ValueError(f"Expected {2}D array for theta and y0, got {len(np.shape(theta))}")

        # # Check that we give the exact number of parameters the model needs
        # if np.shape(theta)[1] != 4 and np.shape(theta)[1] != 10:
        #     raise ValueError(f"Expected either 4 (basic model) or 10 (model with GFP as observable) parameters for the model, got {np.shape(theta)[1]}")

        # # Check that we give the exact number of initial conditions the model needs
        # if np.shape(y0)[1] != 6 and np.shape(y0)[1] != 8:
        #     raise ValueError(f"Expected either 6 (basic model) or 8 (model with GFP as observable) initial conditions for the model, got {np.shape(y0)[1]}")

        
        self.tspan = tspan
        self.tpts = tpts
        self.theta = theta
        self.y0 = y0
        self.teval = np.linspace(tspan[0], tspan[1], tpts)

    # Basic model for the repressilator
    def Repressilator_Basic(self, t: float, y: np.array, *theta):
        mI, mL, mT, pI, pL, pT = y
        alpha_0, alpha, beta, n = theta

        dmIdt = -mI + (alpha/(1+pT**n)) + alpha_0
        dmLdt = -mL + (alpha/(1+pI**n)) + alpha_0
        dmTdt = -mT + (alpha/(1+pL**n)) + alpha_0

        dpIdt = -beta * (pI - mI)
        dpLdt = -beta * (pL - mL)
        dpTdt = -beta * (pT - mT)

        return [dmIdt, dmLdt, dmTdt, dpIdt, dpLdt, dpTdt]

    # Model for the repressilator with GFP as observable
    def Repressilator_GFP(self, t: float, y: np.array, *theta):
        mI, mL, mT, pI, pL, pT,   mG, pG = y
        alpha_0, alpha, beta, n ,   alpha_GFP, kR, kM, n2, d1, d2= theta

        dmIdt = -mI + (alpha/(1+pT**n)) + alpha_0
        dmLdt = -mL + (alpha/(1+pI**n)) + alpha_0
        dmTdt = -mT + (alpha/(1+pL**n)) + alpha_0

        dpIdt = -beta * (pI - mI)
        dpLdt = -beta * (pL - mL)
        dpTdt = -beta * (pT - mT)

        dmGdt = (alpha_GFP/(1+kR*pT**n2)) - d1*mG
        dpGdt = kM*mG - d2*pG

        return [dmIdt, dmLdt, dmTdt, dpIdt, dpLdt, dpTdt,   dmGdt, dpGdt]

    # Simulate repressilator model in batch in a CPU
    def sim(self):

        if np.shape(self.y0)[1] == 6:
            fn = self.Repressilator_Basic
        else:
            fn = self.Repressilator_GFP

        sims = np.zeros((np.shape(self.theta)[0], len(self.teval), np.shape(self.y0)[1]))

        for i in range(np.shape(self.theta)[0]):

            params = self.theta[i,::]
            y00 = self.y0[i,::]
            sol = solve_ivp(
                fun=fn,
                t_span=self.tspan,
                y0=y00,
                t_eval=self.teval,
                args=tuple(params),
                method='BDF'  # Use 'Radau' or 'BDF' if the system is stiff
            )

            for j in range(len(y00)):
                sims[i,::,j] = sol.y[j]

        return sims

            
    # Plot the repressilator model
    def plot(self):
        teval = self.teval
        sims = self.sim()

        if np.shape(sims)[2] == 6:
            fig, ax = plt.subplots(2,1, figsize=(10, 5))
        elif np.shape(sims)[2] == 8:
            fig, ax = plt.subplots(3,1, figsize=(10, 5))


        for i in range(np.shape(sims)[0]):

            lbl_CI = r'$\lambda CI$' if i == 0 else None
            lbl_LacI = r'$LacI$' if i == 0 else None
            lbl_TetR = r'$TetR$' if i == 0 else None
            lbl_GFP1 = r'$mRNA$' if i == 0 else None
            lbl_GFP2 = r'$Protein$' if i == 0 else None

            ax[0].plot(teval, sims[i,::,0], label=lbl_CI, c='#12988d', alpha = 0.4)
            ax[0].plot(teval, sims[i,::,1], label=lbl_LacI, c="#981250", alpha = 0.4)
            ax[0].plot(teval, sims[i,::,2], label=lbl_TetR, c="#8f9812", alpha = 0.4)

            ax[1].plot(teval, sims[i,::,3], label=lbl_CI, c='#12988d', alpha = 0.4)
            ax[1].plot(teval, sims[i,::,4], label=lbl_LacI, c="#981250", alpha = 0.4)
            ax[1].plot(teval, sims[i,::,5], label=lbl_TetR, c="#8f9812", alpha = 0.4)

            if np.shape(sims)[2] == 8:
                ax[2].plot(teval, sims[i,::,6], label=lbl_GFP1, c="#98128D", alpha = 0.4)
                ax[2].plot(teval, sims[i,::,7], label=lbl_GFP2, c="#129819", alpha = 0.4)

        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_ylabel('mRNA')
        ax[0].legend()
        ax[0].set_title('Repressilator')

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        if np.shape(sims)[2] == 6:
            ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Protein')
        ax[1].legend()

        if np.shape(sims)[2] == 8:
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].set_xlabel('Time')
            ax[2].set_ylabel('GFP')
            ax[2].legend()

        
        plt.show()


    def plot2(self, sims):
            teval = self.teval
    
            if np.shape(sims)[2] == 6:
                fig, ax = plt.subplots(2,1, figsize=(10, 5))
            elif np.shape(sims)[2] == 8:
                fig, ax = plt.subplots(3,1, figsize=(10, 5))
    
    
            for i in range(np.shape(sims)[0]):
    
                lbl_CI = r'$\lambda CI$' if i == 0 else None
                lbl_LacI = r'$LacI$' if i == 0 else None
                lbl_TetR = r'$TetR$' if i == 0 else None
                lbl_GFP1 = r'$mRNA$' if i == 0 else None
                lbl_GFP2 = r'$Protein$' if i == 0 else None
    
                ax[0].plot(teval, sims[i,::,0], label=lbl_CI, c='#12988d', alpha = 0.4)
                ax[0].plot(teval, sims[i,::,1], label=lbl_LacI, c="#981250", alpha = 0.4)
                ax[0].plot(teval, sims[i,::,2], label=lbl_TetR, c="#8f9812", alpha = 0.4)
    
                ax[1].plot(teval, sims[i,::,3], label=lbl_CI, c='#12988d', alpha = 0.4)
                ax[1].plot(teval, sims[i,::,4], label=lbl_LacI, c="#981250", alpha = 0.4)
                ax[1].plot(teval, sims[i,::,5], label=lbl_TetR, c="#8f9812", alpha = 0.4)
    
                if np.shape(sims)[2] == 8:
                    ax[2].plot(teval, sims[i,::,6], label=lbl_GFP1, c="#98128D", alpha = 0.4)
                    ax[2].plot(teval, sims[i,::,7], label=lbl_GFP2, c="#129819", alpha = 0.4)
    
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].set_ylabel('mRNA')
            ax[0].legend()
            ax[0].set_title('Repressilator')
    
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            if np.shape(sims)[2] == 6:
                ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Protein')
            ax[1].legend()
    
            if np.shape(sims)[2] == 8:
                ax[2].spines['top'].set_visible(False)
                ax[2].spines['right'].set_visible(False)
                ax[2].set_xlabel('Time')
                ax[2].set_ylabel('GFP')
                ax[2].legend()
    
            
            plt.show()



    # Plot represilator model for interactive plot where you can modify parameter and initial states
    def plotRepress_Single(self, t1,t2,t3,t4, m1,m2,m3,p1,p2,p3):

        tspan = self.tspan
        teval = self.teval
        
        params = [t1,t2,t3,t4]
        y0 = [m1,m2,m3,p1,p2,p3]
        
        sims = np.zeros((1, len(teval), len(y0)))
        
        sol = solve_ivp(
            fun=self.Repressilator_Basic,
            t_span=tspan,
            y0=y0,
            t_eval=teval,
            args=tuple(params),
            method='BDF'  # Use 'Radau' or 'BDF' if the system is stiff
        )


        for j in range(len(y0)):
            sims[0, :, j] = sol.y[j]

        self.plot2(sims)

    # Display interactive plot for plotRepress_Single function
    def play(self):
        
        # 1. Define individual slider widgets
        t1 = widgets.FloatSlider(min=0, max=2, step=1e-5, value=1e-1, readout=True, description=r'alpha_0', layout=Layout(width='400px'))
        t2 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=70, readout=True, description=r'alpha', layout=Layout(width='400px'))
        t3 = widgets.FloatSlider(min=0, max=10, step=1e-5, value=0.32, readout=True, description=r'beta', layout=Layout(width='400px'))
        t4 = widgets.FloatSlider(min=0, max=10, step=1e-5, value=2, readout=True, description=r'n', layout=Layout(width='400px'))

        m1 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=42, readout=True, description=r'mRNA ICI', layout=Layout(width='400px'))
        m2 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=22, readout=True, description=r'mRNA LacI', layout=Layout(width='400px'))
        m3 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=6, readout=True, description=r'mRNA TetR', layout=Layout(width='400px'))

        p1 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=40, readout=True, description=r'Prot ICI', layout=Layout(width='400px'))
        p2 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=15, readout=True, description=r'Prot LacI', layout=Layout(width='400px'))
        p3 = widgets.FloatSlider(min=0, max=100, step=1e-5, value=40, readout=True, description=r'Prot TetR', layout=Layout(width='400px'))



        # 2. Group widgets into two vertical columns (VBox)
        col1 = VBox([t1, t2,  m1,m2,m3])
        col2 = VBox([t3, t4,  p1,p2,p3])

        # 3. Place columns side-by-side in a horizontal box (HBox)
        controls = HBox([col1, col2])

        # 4. Bind widgets and fixed parameters to the callback function
        out = widgets.interactive_output(
            self.plotRepress_Single,
            {
                't1': t1,
                't2': t2,
                't3': t3,
                't4': t4,
                'm1': m1,
                'm2': m2,
                'm3': m3,
                'p1': p1,
                'p2': p2,
                'p3': p3,
            }
        )

        # 5. Display controls on top and the plot output below
        display(controls, out)





