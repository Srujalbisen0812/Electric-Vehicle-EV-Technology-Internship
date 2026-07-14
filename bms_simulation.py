import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BatteryManagementSystem:
    def __init__(self):
        self.capacity=100.0
        self.soc=100.0
        self.soh=100.0
        self.temperature=25.0
        self.ambient_temp=25.0
        self.current=40.0
        self.voltage=400.0
        self.internal_resistance=0.05
        self.mode="Discharge"
        self.cycles=0
        self.R_th=2.5
        self.C_th=500
        self.cells=np.random.normal(3.70,0.02,8)
        self.time=[]; self.soc_history=[]; self.soh_history=[]; self.temp_history=[]
    def estimate_soc(self):
        dt=1/3600
        if self.mode=="Discharge":
            self.soc-=(self.current*dt/self.capacity)*100
        else:
            self.soc+=(self.current*dt/self.capacity)*100
        self.soc=np.clip(self.soc,0,100)
    def estimate_soh(self):
        cycle_deg=self.cycles*0.035
        temp_stress=max(0,self.temperature-40)*0.003
        self.soh=max(70,100-cycle_deg-temp_stress)
    def update_voltage(self):
        ocv=3.2+0.8*(self.soc/100)+0.3*((self.soc/100)**2)
        self.cells=np.random.normal(ocv,0.015,8)
        self.voltage=np.sum(self.cells)
    def thermal_model(self):
        q=(self.current**2)*self.internal_resistance
        loss=(self.temperature-self.ambient_temp)/self.R_th
        cool=100 if self.temperature>45 else 0
        self.temperature+=(q-loss-cool)/self.C_th
    def balance_cells(self):
        if np.max(self.cells)-np.min(self.cells)>0.02:
            print("Passive Cell Balancing Active")
    def alerts(self):
        if self.temperature>60: print("Over Temperature!")
        if np.max(self.cells)>4.2: print("Cell Over Voltage!")
        if self.soc<15: print("Low Battery!")
    def step(self):
        self.estimate_soc(); self.estimate_soh(); self.update_voltage()
        self.thermal_model(); self.balance_cells(); self.alerts()
        self.time.append(len(self.time))
        self.soc_history.append(self.soc)
        self.soh_history.append(self.soh)
        self.temp_history.append(self.temperature)

bms=BatteryManagementSystem()
fig,ax=plt.subplots(3,1,figsize=(10,8))
def animate(i):
    bms.step()
    for a in ax: a.cla()
    ax[0].plot(bms.time,bms.soc_history)
    ax[0].set_title("SOC (%)"); ax[0].set_ylim(0,100)
    ax[1].plot(bms.time,bms.soh_history)
    ax[1].set_title("SOH (%)"); ax[1].set_ylim(70,100)
    ax[2].plot(bms.time,bms.temp_history)
    ax[2].set_title("Temperature (°C)")
    plt.tight_layout()
ani=FuncAnimation(fig,animate,interval=200)
plt.show()
