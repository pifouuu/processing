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
    'figure.figsize': [9, 8]
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

dirnames_tb1= [
"02-05-2017_20-37-56_2_v_0.000000_n_0.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"04-05-2017_21-02-49_2_v_0.000000_n_0.100000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_21-48-56_2_v_0.000000_n_0.500000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_22-34-12_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_23-20-37_2_v_0.000000_n_5.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"05-05-2017_00-08-34_2_v_0.000000_n_10.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"03-05-2017_03-27-22_2_v_0.000000_n_20.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_2001_size_5",
"10-05-2017_15-23-20_2_v_0.0_n_30.0_tb_1.0_explo_0_tutorAtt_10000_pretrain_0_fR_100_nmodels_5_batch_1_steps_1001_size_5"
]

dirnames_selected = [
"02-05-2017_18-32-32_2_v_0.000000_n_0.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"02-05-2017_20-37-56_2_v_0.000000_n_0.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
#"05-05-2017_01-46-52_2_v_0.000000_n_0.000000_tb_5.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"02-05-2017_21-37-43_2_v_0.000000_n_1.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"04-05-2017_22-34-12_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"02-05-2017_22-16-57_2_v_0.000000_n_5.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5",
"04-05-2017_23-20-37_2_v_0.000000_n_5.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"04-05-2017_17-56-08_2_v_0.000000_n_10.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5",
"05-05-2017_00-08-34_2_v_0.000000_n_10.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_801_size_5"
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

def load(dir, nb_r, nb_i):
    path_list = glob.glob(dir+'/*/accumulated_rewards')
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

data = [load(data_dir, min(param_nb, nb_iter)/25, nb_iter/25)[:,nb_iter/25-1] for (data_dir,param_nb, param_n, param_tb, param_explo, param_tutorAtt) in params
        if (True
            # and param_tb == 1.
            # and param_n == 0.1
            # and param_explo == 500
            # and param_tutorAtt == nb_iter
            )]
data_params = [(param_n, param_tb, param_explo, param_tutorAtt) for (_, _, param_n, param_tb, param_explo, param_tutorAtt) in params
               if (True
                   # and param_tb == 1.
                   # and param_n == 0.1
                   # and param_explo == 500
                   # and param_tutorAtt == nb_iter
                   )]

fig = plt.figure()
fig.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, wspace=0.1)
ax2 = fig.add_subplot(111)

meanpointprops = dict(marker='o', markeredgecolor='black',
                      markerfacecolor='firebrick')
bp = ax2.boxplot(data, notch=0, sym='b+', vert=1, whis=1.5,
             positions=None, widths=0.3, showmeans=True, meanprops=meanpointprops)

ax2.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax2.set_axisbelow(True)


ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', length=0)
ax2.set_ylim(-100, 7500)
ax2.set_yticks(np.arange(0, 7500, 1000))
ax2.set_xlabel("Hyperparameter n & t")
ax2.set_ylabel("Reward accumulated at step 800")

for spine in ax2.spines.values():
    spine.set_position(('outward', 5))



ax2.set_xticklabels(["n: {:.1f} \n t: {:.1f}".format(param_n, param_tb) for (param_n, param_tb, param_explo, param_tutorAtt) in
                     data_params])


for i in range(0, len(bp['boxes'])):
   bp['boxes'][i].set_color(colors[i])
   # we have two whiskers!
   bp['whiskers'][i*2].set_color(colors[i])
   bp['whiskers'][i*2 + 1].set_color(colors[i])
   bp['whiskers'][i*2].set_linewidth(1)
   bp['whiskers'][i*2 + 1].set_linewidth(1)
   bp['whiskers'][i * 2].set_linestyle('solid')
   bp['whiskers'][i * 2 + 1].set_linestyle('solid')
   # top and bottom fliers
   # (set allows us to set many parameters at once)
   for flier in bp['fliers']:
       flier.set(marker='+', markersize=6)
   bp['medians'][i].set_color('black')
   bp['medians'][i].set_linewidth(1)
   # and 4 caps to remove
   for c in bp['caps']:
       c.set_linewidth(1)

# Create custom artists
outlierArtist = plt.Line2D([0], [0], linestyle='none', color='black',
                          marker='+')
medianArtist = plt.Line2D((0, 1), (0, 0), color='k')
avgArtist = plt.Line2D([0], [0], linestyle='none', marker='o', markeredgecolor='black',
                      markerfacecolor='firebrick')

# Create legend from custom artist/label lists
ax2.legend([medianArtist, avgArtist, outlierArtist], ['Median', 'Average', 'Outliers'], numpoints=1, ncol=3, loc=9)

stars_list = []
for i in range(1,len(data)-1,2):
    z, p = scipy.stats.mannwhitneyu(data[i], data[i+1])
    p_value = p * 2
    s = stars(p_value)
    stars_list.append(s)


for i, star in enumerate(stars_list):
    ax2.annotate("", xy=(2*i+2, 6100), xycoords='data',
               xytext=(2*i+3, 6100), textcoords='data',
               arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                               connectionstyle="bar,fraction=0.1"))
    ax2.text(2*i+2.5, 6400, star,
           horizontalalignment='center',
           verticalalignment='center',
            fontsize='xx-large')


fig.savefig("/home/pierre/Dropbox/images/boxplotsNormal.png")
plt.show()


