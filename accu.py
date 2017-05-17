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
    'figure.figsize': [9, 7]
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

dirnames_selected = [
#"02-05-2017_18-32-32_2_v_0.000000_n_0.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"02-05-2017_20-37-56_2_v_0.000000_n_0.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"05-05-2017_01-46-52_2_v_0.000000_n_0.000000_tb_5.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
#"02-05-2017_21-37-43_2_v_0.000000_n_1.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"04-05-2017_22-34-12_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"02-05-2017_22-16-57_2_v_0.000000_n_5.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"04-05-2017_23-20-37_2_v_0.000000_n_5.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
#"04-05-2017_17-56-08_2_v_0.000000_n_10.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
#"05-05-2017_00-08-34_2_v_0.000000_n_10.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5"
]

nb_iter = 800

params_split = [name.split("_") for name in dirnames_selected]
params_nb = [int(param_split[-3]) for param_split in params_split]
params_n = [float(param_split[6]) for param_split in params_split]
params_tb = [float(param_split[8]) for param_split in params_split]
params_explo = [int(param_split[10]) if param_split[9]=="explo" else 0 for param_split in params_split]
params_tutorAtt = [int(param_split[12]) if param_split[11]=="tutorAtt" else nb_iter for param_split in params_split]

data_dirs = [dirDrop+dirname for dirname in dirnames_selected]
params = zip(data_dirs, params_nb, params_n, params_tb, params_explo, params_tutorAtt)

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

data = [load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "accumulated_rewards") for (data_dir,param_nb, param_n, param_tb, param_explo, param_tutorAtt) in params
        if (True
            #and param_tb == 1.
            #and param_n == 0.1
            #and param_explo == 500
            #and param_tutorAtt == nb_iter
            )]

data_tutor = [load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "accumulated_tutor_rewards") for (data_dir,param_nb, param_n, param_tb, param_explo, param_tutorAtt) in params
              if (True
                  # and param_tb == 1.
                  # and param_n == 0.1
                  # and param_explo == 500
                  # and param_tutorAtt == nb_iter
                  )]

data_params = [(param_n, param_tb, param_explo, param_tutorAtt) for (_, _, param_n, param_tb, param_explo, param_tutorAtt) in params
               if (True
                   #and float(param_tb) == 1.
                   #and float(param_n) == 0.1
                   #and int(param_explo) == 500
                   #and param_tutorAtt == "inf"
                   )]
data_perc = [perc(datum) for datum in data]
data_avg = [avg(datum) for datum in data]
data_tutor_avg = [avg(datum) for datum in data_tutor]
data_tutor_perc = [perc(datum) for datum in data_tutor]

data_both = zip(data_avg,data_tutor_avg)



fig = plt.figure()
ax1 = fig.add_subplot(111)

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
ax1.set_xlim(0, 810)
#ax1.set_ylim(0,1)
ax1.set_ylim(0, 4500)
ax1.set_xlabel("Step")
ax1.set_ylabel("Accumulated tutor reward")
ax1.set_xticks(np.arange(0, 810, 200))
#ax1.set_yticks(np.arange(0,1,0.1))
ax1.set_yticks(np.arange(0, 4500, 1000))

dash_styles = ['--', '-', '-.', ':' ]

# for i,(data_item,data_tutor_item) in enumerate(data_both):
#     (param_n, param_tb, param_explo, param_tutorAtt) = data_params[i]
#     if param_tb==1.:
#         l = 'solid'
#     if param_tb!=1.:
#         l = 'dashed'
#     if param_n==0.:
#         l = '-.'
#         c = colors[0]
#     if param_n==1.:
#         c=colors[1]
#     if param_n==5.:
#         c=colors[2]
#     if param_n==10.:
#         c=colors[3]
#     ax1.plot(x, exp_smooth(data_tutor_item.tolist(),0.9), linewidth=2, color=c, linestyle=l)
#     #ax1.plot(x, data_tutor_item/data_item, linewidth=2, color=colors[i])

for i,data_item in enumerate(data_tutor_perc):
    ax1.fill_between(x, data_item[1], data_item[2], alpha=0.25, linewidth=0, color=colors[i])
    ax1.plot(x, data_item[0].tolist(), linewidth=2, linestyle='solid', color=colors[i],
             label='n: {:.1f}, t: {:.1f}, median'.format(data_params[i][0], data_params[i][1]))
    ax1.plot(x,data_tutor_avg[i], linewidth=2, linestyle='--', color=colors[i],
             label='n: {:.1f}, t: {:.1f}, average'.format(data_params[i][0], data_params[i][1]))
ax1.legend()
legend = ax1.legend(loc=2);

# ax1.legend()
# legend = ax1.legend(["n: {:.1f}, t: {:.1f}".format(param_n,param_tb)
#                      for (param_n, param_tb, param_explo, param_tutorAtt) in data_params], loc=2);
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('0.0')
fig.savefig("/home/pierre/Dropbox/images/comparaisonNormaleTutor.png")

plt.show()