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

test = [
"12-05-2017_15-02-33_v_0.0_n_5.0_tb_1.0_startR_500_tutorAtt_100_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18"
]
dirnames_selected = [
# dirDrop+"10-05-2017_18-02-49_v_0.0_n_1.0_tb_1.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
# dirDrop+"10-05-2017_19-32-37_v_0.0_n_1.0_tb_0.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
# dirDrop+"10-05-2017_21-00-47_v_0.0_n_5.0_tb_1.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
# dirDrop+"10-05-2017_22-30-17_v_0.0_n_5.0_tb_0.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
# dirDrop+"10-05-2017_23-57-48_v_0.0_n_10.0_tb_0.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
# dirDrop+"11-05-2017_01-25-47_v_0.0_n_10.0_tb_1.0_startR_500_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5",
dirLoc+"15-05-2017_17-58-11_v_0.0_n_1.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"15-05-2017_19-29-42_v_0.0_n_5.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"15-05-2017_21-00-50_v_0.0_n_10.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"15-05-2017_22-31-43_v_0.0_n_1.0_tb_1.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"16-05-2017_00-02-56_v_0.0_n_5.0_tb_1.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"16-05-2017_01-33-34_v_0.0_n_10.0_tb_1.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_1601",
dirLoc+"16-05-2017_03-05-08_v_0.0_n_1.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_500",
dirLoc+"16-05-2017_04-36-21_v_0.0_n_5.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_500",
dirLoc+"16-05-2017_06-06-26_v_0.0_n_10.0_tb_0.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_500",
dirLoc+"16-05-2017_07-37-00_v_0.0_n_1.0_tb_1.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_500",
dirLoc+"16-05-2017_09-08-49_v_0.0_n_5.0_tb_1.0_startR_500_tutorAtt_1601_pretrain_0_fR_100_nmodels_5_batch_1_steps_1601_size_5_evalExp_18_evalStep_500_resetPlan_500"
]

nb_iter = 1600

params_split = [name.split("_") for name in dirnames_selected]
params_nb = [int(param_split[param_split.index('steps')+1]) for param_split in params_split]
params_n = [float(param_split[param_split.index('n')+1]) for param_split in params_split]
params_tb = [float(param_split[param_split.index('tb')+1]) for param_split in params_split]
params_startR = [int(param_split[param_split.index('startR')+1]) for param_split in params_split]
params_reset = [int(param_split[param_split.index('resetPlan')+1]) for param_split in params_split]

data_dirs = [dirname for dirname in dirnames_selected]
params = zip(data_dirs, params_nb, params_n, params_tb, params_startR, params_reset)

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

data = [load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "accumulated_rewards_pick_red") for (data_dir,param_nb, param_n, param_tb, param_startR, param_reset) in params
        if (True
            #and param_tb == 1.
            #and param_n == 0.1
            #and param_explo == 500
            #and param_tutorAtt == nb_iter
            )]

data_tutor = [load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25, "accumulated_tutor_rewards") for (data_dir,param_nb, param_n, param_tb, param_startR, param_reset) in params
              if (True
                  # and param_tb == 1.
                  # and param_n == 0.1
                  # and param_explo == 500
                  # and param_tutorAtt == nb_iter
                  )]

data_params = [(param_n, param_tb, param_startR, param_reset) for (_, _, param_n, param_tb, param_startR, param_reset) in params
               if (True
                   #and float(param_tb) == 1.
                   #and float(param_n) == 0.1
                   #and int(param_explo) == 500
                   #and param_tutorAtt == "inf"
                   )]
data_perc = [perc(datum) for datum in data]
data_avg = [avg(datum) for datum in data]
data_tutor_avg = [avg(datum) for datum in data_tutor]
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
ax1.set_xlim(0, 1610)
#ax1.set_ylim(0,1)
ax1.set_ylim(0, 80)
ax1.set_xlabel("Step")
ax1.set_ylabel("Accumulated reward")
ax1.set_xticks(np.arange(0, 1610, 200))
#ax1.set_yticks(np.arange(0,1,0.1))
ax1.set_yticks(np.arange(0, 80, 10))

dash_styles = ['--', '-', '-.', ':' ]

for i,(data_item,data_tutor_item) in enumerate(data_both):
    corrected = [elem - data_item[20] for elem in data_item[20:]]
    (param_n, param_tb, param_startR, param_reset) = data_params[i]
    if param_tb == 1.:
        l = 'solid'
    if param_tb != 1.:
        l = 'dashed'
    if param_n == 0.:
        l = '-.'
        c = colors[0]
    if param_n == 1.:
        c = colors[1]
    if param_n == 5.:
        c = colors[2]
    if param_n == 10.:
        c = colors[3]
    if param_reset == 500:
        s = 'o'
    if param_reset == 1601:
        s = 'd'
    #ax1.plot(x[20:], exp_smooth(corrected,0.9), linewidth=2, color=c, linestyle=l)
    ax1.plot(x, exp_smooth(data_item,0.9), linewidth=2, color=c, linestyle=l, marker=s, markersize = 4)

    #ax1.plot(x, data_tutor_item/data_item, linewidth=2, color=colors[i])

#for i,data_item in enumerate(data_perc):
    #ax1.fill_between(x, data_item[1], data_item[2], alpha=0.25, linewidth=0, color=colors[i])
    #ax1.plot(x, data_item[0].tolist(), linewidth=2, linestyle='--', color=colors[i])

ax1.legend()
legend = ax1.legend(["n: {:.1f}, t: {:.1f}".format(param_n,param_tb)
                     for (param_n, param_tb, param_startR, param_reset) in data_params], loc=2);
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('0.0')
fig.savefig("/home/pierre/Dropbox/images/comparaisonExlo.png")

plt.show()