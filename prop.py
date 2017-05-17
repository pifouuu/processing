from __future__ import division

import numpy as np
from sys import argv
import struct
import matplotlib.pyplot as plt
import glob
import brewer2mpl
import scipy.stats



# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors
params = {
    'axes.labelsize': 16,
    'text.fontsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [15, 8]
}
plt.rcParams.update(params)

dirDrop = "/home/pierre/Dropbox/resultats/resultats_python/"
dirLoc = "/home/pierre/workspace/myTexplore/resultats_python/"

def read(path):
    var = []
    with open(path) as f:
        while True:
            data = f.read(4)
            if not data: break
            s, = struct.unpack('f', data)
            var.append(s)
    return np.array(var)



def perc(data):
   median = np.zeros(data.shape[1])
   perc_25 = np.zeros(data.shape[1])
   perc_75 = np.zeros(data.shape[1])
   for i in range(0, len(median)):
       median[i] = np.median(data[:, i])
       perc_25[i] = np.percentile(data[:, i], 25)
       perc_75[i] = np.percentile(data[:, i], 75)
   return median, perc_25, perc_75

def avg(data):
    avg = np.zeros(data.shape[1])
    for i in range(0, len(avg)):
        avg[i] = np.average(data[:,i])
    return avg

def stars(p):
   if p < 0.0001:
       return "****"
   elif (p < 0.001):
       return "***"
   elif (p < 0.01):
       return "**"
   elif (p < 0.05):
       return "*"
   else:
       return "-"

dirnames = [
#"02-05-2017_17-25-42_2_v_0.000000_n_0.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_0_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_18-07-41_2_v_0.000000_n_10.000000_tb_10.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_18-32-32_2_v_0.000000_n_0.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_20-11-54_2_v_0.000000_n_0.500000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_20-37-56_2_v_0.000000_n_0.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_21-14-47_2_v_0.000000_n_0.100000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_21-37-43_2_v_0.000000_n_1.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"02-05-2017_22-16-57_2_v_0.000000_n_5.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"02-05-2017_22-39-22_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"03-05-2017_03-27-22_2_v_0.000000_n_20.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_2001_size_5",
"04-05-2017_17-56-08_2_v_0.000000_n_10.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_18-43-35_2_v_0.000000_n_20.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_19-30-18_2_v_0.000000_n_2.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_21-02-49_2_v_0.000000_n_0.100000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_21-48-56_2_v_0.000000_n_0.500000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_22-34-12_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_23-20-37_2_v_0.000000_n_5.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"05-05-2017_00-08-34_2_v_0.000000_n_10.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
#"05-05-2017_01-46-52_2_v_0.000000_n_0.000000_tb_5.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"05-05-2017_02-31-21_2_v_0.000000_n_0.100000_tb_5.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"05-05-2017_18-53-01_2_v_0.000000_n_0.100000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"05-05-2017_19-51-30_2_v_0.000000_n_0.500000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"05-05-2017_20-49-43_2_v_0.000000_n_1.000000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"05-05-2017_21-47-49_2_v_0.000000_n_2.000000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"05-05-2017_22-46-08_2_v_0.000000_n_10.000000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"05-05-2017_23-44-40_2_v_0.000000_n_0.100000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_00-41-36_2_v_0.000000_n_0.500000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_01-37-58_2_v_0.000000_n_1.000000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_02-34-42_2_v_0.000000_n_2.000000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_03-33-18_2_v_0.000000_n_5.000000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_04-33-32_2_v_0.000000_n_10.000000_tb_1.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"06-05-2017_05-32-15_2_v_0.000000_n_5.000000_tb_0.000000_explo_500_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5"
]


dirname = "04-05-2017_21-48-56_2_v_0.000000_n_0.500000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5"
data_dir = dirDrop+dirname
param_split = dirname.split("_")
param_nb = int(param_split[-3])
param_n = param_split[6]
param_tb = param_split[8]
param_explo = param_split[10] if param_split[9]=="explo" else "0"
param_tutorAtt = param_split[12] if param_split[11]=="tutorAtt" else "inf"
param = (data_dir, param_nb, param_n, param_tb, param_explo, param_tutorAtt)

nb_iter = 800
x = np.linspace(0,nb_iter,nb_iter/25)

def load(dir, nb_r, nb_i, target):
    path_list = glob.glob(dir+'/*/'+target)
    data = np.zeros((len(path_list), nb_i))
    for i,p in enumerate(path_list):
        readings = read(p)[:nb_r]
        data[i,:nb_r] = readings
    return data

def derive(tab):
    res = [(tab[i+1]-tab[i]) for i in range(len(tab)-1)]
    max_val = 25/12*100
    return [l/max_val for l in res]

def exp_smooth(tab, alpha):
    smooth = [tab[0]]
    for i in range(len(tab)-1):
        smooth.append(alpha*tab[1+i]+(1-alpha)*smooth[i])
    return smooth

data = [
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "reward_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "sync_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "nov_prop")
    ]
data_perc = [perc(datum) for datum in data]
data_avg = [avg(datum) for datum in data]




fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(0, 800)
ax.set_xlabel("Step")
ax.xaxis.set_label_coords(0.5,-0.07)
ax.set_ylabel("Proportions")
ax.set_xticks([])
ax.set_yticks([])
ax.yaxis.set_label_coords(-0.05,0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax1 = fig.add_subplot(131)

ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.tick_params(axis='x', direction='out')
ax1.tick_params(axis='y', length=0)
for spine in ax1.spines.values():
    spine.set_position(('outward', 5))
ax1.set_ylim(-0.1, 1)
ax1.set_yticks(np.arange(0,1,0.1))




for i,data_item in enumerate(data_avg):
    tab = data_item.tolist()[1:]
    tab = [25*elem for elem in tab]
    ax1.plot(x[1:], exp_smooth(tab,0.2), linewidth=2, color=colors[i])

ax1.text(.5,.03,"n: {:.1f}, t: {:.1f}".format(0.5,1),
        horizontalalignment='center',
        transform=ax1.transAxes)

ax1.legend()
legend = ax1.legend(["External reward", "Synchrony", "Novelty"], loc=1);
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('0.0')

ax2 = fig.add_subplot(132)
data_dir = dirDrop + \
           "05-05-2017_00-08-34_2_v_0.000000_n_10.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5"

data = [
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "reward_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "sync_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "nov_prop")
    ]
data_perc = [perc(datum) for datum in data]
data_avg = [avg(datum) for datum in data]

ax2.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', length=0)
for spine in ax2.spines.values():
    spine.set_position(('outward', 5))
# ax2.set_xlim(0, 800)
# ax2.set_ylim(0, 1)
# ax2.set_xticks(np.arange(0, 800, 200))
ax2.set_ylim(-0.1, 1)
ax2.set_yticks(np.arange(0,1,0.1))
ax2.set_yticklabels([])


for i,data_item in enumerate(data_avg):
    tab = data_item.tolist()[1:]
    tab = [25*elem for elem in tab]
    ax2.plot(x[1:], exp_smooth(tab,0.2), linewidth=2, color=colors[i])

ax2.text(.5,.03,"n: {:.1f}, t: {:.1f}".format(10,1),
        horizontalalignment='center',
        transform=ax2.transAxes)

ax3 = fig.add_subplot(133)
data_dir = dirDrop + \
           "10-05-2017_15-23-20_2_v_0.0_n_30.0_tb_1.0_explo_0_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1001_size_5"

data = [
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "reward_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "sync_prop"),
    load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "nov_prop")
    ]
data_perc = [perc(datum) for datum in data]
data_avg = [avg(datum) for datum in data]

ax3.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.get_xaxis().tick_bottom()
ax3.get_yaxis().tick_left()
ax3.tick_params(axis='x', direction='out')
ax3.tick_params(axis='y', length=0)
for spine in ax3.spines.values():
    spine.set_position(('outward', 5))
# ax3.set_xlim(0, 800)
# ax3.set_ylim(0, 1)
# ax3.set_xticks(np.arange(0, 800, 200))
ax3.set_ylim(-0.1, 1)
ax3.set_yticks(np.arange(0,1,0.1))
ax3.set_yticklabels([])

for i,data_item in enumerate(data_avg):
    tab = data_item.tolist()[1:]
    #tab = [25*elem for elem in tab]
    ax3.plot(x[1:], exp_smooth(tab,0.2), linewidth=2, color=colors[i])

ax3.text(.5,.03,"n: {:.1f}, t: {:.1f}".format(30,1),
        horizontalalignment='center',
        transform=ax3.transAxes)

fig.savefig("/home/pierre/Dropbox/images/prop.png")
plt.show()