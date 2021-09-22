from datetime import datetime

from deepSI.systems.double_pendulum import DoublePendulum, normalize_double_pendulum, double_pendulum_to_video
from deepSI.system_data import System_data

from deepSI.fit_systems.LNN import LNN
from deepSI.fit_systems.LNN import LNN_system

import numpy as np

now = datetime.now()
datetimestr = now.strftime("%Y%m%d_%H_%M_%S")
print("****************************")
print("Experiment started: {}".format(datetimestr))
print("****************************")
print(" ")
print(" ")



sys = DoublePendulum(x0=np.array([3*np.pi/7, 3*np.pi/4, 0, 0], dtype=np.float32))
exp = System_data(u=0.2*np.ones(shape=(1500, 4), dtype=np.float32))
train_data = sys.apply_experiment(exp, save_state=True)

train_clip = double_pendulum_to_video(train_data)
train_clip.write_videofile('pendulum_train.mp4')

exp_test = System_data(u=0.2*np.ones(shape=(302, 4), dtype=np.float32))
sys2 = DoublePendulum(x0=np.array([2*np.pi/7, 2*np.pi/4, 0, 0], dtype=np.float32))
test_data = sys2.apply_experiment(exp_test, save_state=True)

test_clip = double_pendulum_to_video(test_data)
test_clip.write_videofile('pendulum_test.mp4')


from deepSI.systems.double_pendulum import normalize_double_pendulum


lnn = LNN_system(normalization_fn=normalize_double_pendulum, nu=4, nx=4, ny=2, dt=0.01, x0=np.array([2*np.pi/7, 2*np.pi/4, 0, 0], dtype=np.float32))

lnn.fit_and_validate(train_data, test_data=test_data, num_batches=1500)
lnn.save_params('Params_{}.npy'.format(datetimestr))
#lnn.load_params('Params_20210715_13_26_24.npy')


simulation_data = lnn.apply_experiment(exp_test, save_state=True)
simulation_clip = double_pendulum_to_video(simulation_data)
simulation_clip.write_videofile('simulation_pendulum.mp4')


