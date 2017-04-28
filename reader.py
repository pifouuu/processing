import numpy as np
from sys import argv
import struct
import matplotlib.pyplot as plt
import glob
import brewer2mpl

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

variable = str(argv[1])
path1 = str(argv[2])
path2 = str(argv[3])
dir = "/home/pierre/workspace/myTexplore/resultats_python/"

paths_var = []
paths_avg = []

def read(path):
    var = []
    with open(path) as f:
        while True:
            data = f.read(4)
            if not data: break
            s, = struct.unpack('f', data)
            var.append(s)
    return np.array(var)

def load(dir):
    path_list = glob.glob(dir+'/*/accumulated_rewards')
    data = np.zeros((len(path_list), 41))
    for i,p in enumerate(path_list):
        data[i,:] = read(p)
    return data

def med(data):
    median = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
    return median

data = load(dir+"28-04-2017_16-28-48_2_v_0.000000_n_0.000000_tb_1.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_1_batch_1_steps_1001_size_5")
print(data)
med_data = med(data)
print(med_data)
# for arg in argv[2:]:
#     paths_var.append(dir+str(arg)+"/"+variable)
#     paths_avg.append(dir+str(arg)+"/step_reached")
#
# for (path_var,path_avg) in zip(paths_var,paths_avg):
#
#     params = path_var.split("/")[-2]
#     idx_tb_val = params.find("tb_")
#     tb_val = float(params[idx_tb_val:].split("_")[1])
#
#     var = []
#     with open(path_var) as fvar:
#         while True:
#             data = fvar.read(4)
#             if not data: break
#             s, = struct.unpack('f', data)
#             var.append(s)
#
#     avg = []
#     with open(path_avg) as favg:
#         while True:
#             data = favg.read(4)
#             if not data: break
#             s, = struct.unpack('f', data)
#             avg.append(s)
#
#     x = np.linspace(0,1000,41)
#
#     plt.plot(x,[v/a if a != 0 else v for (v,a) in zip(var,avg)],
#              label='tutor bonus : {:2.1f}'.format(tb_val))
#     plt.legend()
#
# plt.show()