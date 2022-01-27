from datetime import datetime

from deepSI.systems.double_pendulum import DoublePendulum, normalize_double_pendulum, double_pendulum_video, double_pendulum_plot
from deepSI.system_data import System_data
import matplotlib.pyplot as plt
from deepSI.fit_systems.LNN import LNN
from deepSI.fit_systems.LNN import LNN_system
from deepSI.system_data import load_system_data

import numpy as np

now = datetime.now()
datetimestr = now.strftime("%Y%m%d_%H_%M_%S")
print("****************************")
print("Experiment started: {}".format(datetimestr))
print("****************************")
print(" ")
print(" ")


print('Generate training data')

# Create a double pendulum to generate data with given initial condition and input
sys = DoublePendulum(x0=np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32))
#
# sim = load_system_data('HNN_Simulation_data_20211116_23_51_00.npz')
# test = load_system_data('Test_data_20211117_01_00_25.npz')

# sim = load_system_data('Simulation_data_20211117_01_00_25.npz')
# test = load_system_data('Test_data_20211117_01_00_25.npz')

sim = load_system_data('SSE_Simulation_data_20211118_09_38_44.npz')
test = load_system_data('SSE_Test_data_20211118_09_38_44.npz')

offset=475
sim = System_data(y=sim.y[offset:][:])
test = System_data(y=test.y[:-offset, :][:])

# sim = np.array([[1,2,3,4], [5,6,7,8]])
# test = np.array([[1,2,3,7], [5,6,7,8]])

def nrmse(x1, x2):

    # print(x1)
    # print(x2)
    diff = x1 - x2
    # print(diff)
    se = np.power(diff, 2)
    # print(se)
    mse = se.mean()
    # print(mse)
    rmse = np.sqrt(mse)
    # print(rmse)
    range = np.max(x2) - np.min(x2)
    # print(range)
    nrmse = rmse / range
    # print(nrmse)
    return nrmse

nrmse_t1 = nrmse(sim.y[:, 0], test.y[:, 0])
nrmse_t2 = nrmse(sim.y[:, 1], test.y[:, 1])
nrmse_w1 = nrmse(sim.y[:, 2], test.y[:, 2])
nrmse_w2 = nrmse(sim.y[:, 3], test.y[:, 3])

ratio = 1500/(1500-offset)
print(nrmse_t1*ratio)
print(nrmse_t2*ratio)
print(nrmse_w1*ratio)
print(nrmse_w2*ratio)

sum = nrmse_t1 + nrmse_t2 + nrmse_w1 + nrmse_w2
print(sum)
print(sim.NRMS(test))

# fig = double_pendulum_plot(0, sim, test, sys)
# plt.show()
# vid = double_pendulum_video(sim, test, sys)
# vid.write_videofile('SSE_final.mp4')




