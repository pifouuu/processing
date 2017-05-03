import numpy as np
from sys import argv
import struct
import matplotlib.pyplot as plt
import glob
import brewer2mpl
import scipy.stats


# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

dir = "/home/pierre/workspace/myTexplore/resultats_python/"


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
        data[i,:] = read(p)[:41]
    return data

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

data_baseline = load(dir+"02-05-2017_21-37-43_2_v_0.000000_n_1.000000_tb_0.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5")
med_baseline, perc_25_baseline, perc_75_baseline = perc(data_baseline)
avg_data_tb0 = avg(data_baseline)

data_comp = load(dir+"02-05-2017_22-39-22_2_v_0.000000_n_1.000000_tb_1.000000_eps_0.000000_pretrain_0_fR_100_nbR_2_nbB_1_nmodels_5_batch_1_steps_1001_size_5")
med_comp, perc_25_comp, perc_75_comp = perc(data_comp)
avg_data_tb1 = avg(data_comp)

x = np.linspace(0,1000,41)

fig = plt.figure()
fig.subplots_adjust(left=0.09, bottom=0.1, right=0.99, top=0.99, wspace=0.1)
ax1 = fig.add_subplot(121)

params = {
    'axes.labelsize': 8,
    'text.fontsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
plt.rcParams.update(params)

ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax1.fill_between(x, perc_25_baseline, perc_75_baseline, alpha=0.25, linewidth=0, color=colors[0])
ax1.plot(x,med_baseline, linewidth=2, color=colors[0])
ax1.plot(x,avg_data_tb0, linewidth=2, linestyle='--', color=colors[0])

ax1.fill_between(x, perc_25_comp, perc_75_comp, alpha=0.25, linewidth=0, color=colors[1])
ax1.plot(x,med_comp, linewidth=2, color=colors[1])
ax1.plot(x,avg_data_tb1, linewidth=2, linestyle='--', color=colors[1])

ax1.legend()

ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 10000)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.tick_params(axis='x', direction='out')
ax1.tick_params(axis='y', length=0)

for spine in ax1.spines.values():
    spine.set_position(('outward', 5))

ax1.set_xticks(np.arange(0, 1000, 200))

legend = ax1.legend(["Median, tb = 0", "Average, tb = 0", "Median, tb = 1", "Average, tb = 1"], loc=0);
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('1.0')


ax2 = fig.add_subplot(122)

data_1000_tb0 = data_baseline[:,40]
data_1000_tb1 = data_comp[:,40]

z, p = scipy.stats.mannwhitneyu(data_1000_tb0, data_1000_tb1)
p_value = p * 2
s = stars(p)

y_max = np.max(np.concatenate((data_1000_tb0, data_1000_tb1)))
y_min = np.min(np.concatenate((data_1000_tb0, data_1000_tb1)))

ax2.set_ylim(0, 10000)
ax2.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax2.set_axisbelow(True)


ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.tick_params(axis='x', direction='out')
ax2.tick_params(axis='y', length=0)

for spine in ax2.spines.values():
    spine.set_position(('outward', 5))


bp = ax2.boxplot([data_1000_tb0, data_1000_tb1], notch=0, sym='b+', vert=1, whis=1.5,
             positions=None, widths=0.3)

ax2.set_yticklabels([])
ax2.set_xticklabels(['tb = 0', 'tb = 1'])


for i in range(0, len(bp['boxes'])):
   bp['boxes'][i].set_color(colors[i])
   # we have two whiskers!
   bp['whiskers'][i*2].set_color(colors[i])
   bp['whiskers'][i*2 + 1].set_color(colors[i])
   bp['whiskers'][i*2].set_linewidth(2)
   bp['whiskers'][i*2 + 1].set_linewidth(2)
   # top and bottom fliers
   # (set allows us to set many parameters at once)
   for flier in bp['fliers']:
       flier.set(markerfacecolor='black',
                   marker='o', alpha=0.75, markersize=6,
                   markeredgecolor='none')
   bp['medians'][i].set_color('black')
   bp['medians'][i].set_linewidth(3)
   # and 4 caps to remove
   for c in bp['caps']:
       c.set_linewidth(0)

ax2.annotate("", xy=(1, 1.05*y_max), xycoords='data',
           xytext=(2, 1.05*y_max), textcoords='data',
           arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                           connectionstyle="bar,fraction=0.1"))
ax2.text(1.5, 1.05*y_max + abs(y_max - y_min)*0.1, stars(p_value),
       horizontalalignment='center',
       verticalalignment='center',
        fontsize='xx-large')
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
plt.show()