#%% imports and settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UserPackage_AiryDiskModel import AiryModel
import matplotlib.colors as mcolors
import matplotlib.cm as cm
fontsize = 12
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize

hot_cmap = cm.get_cmap('hot')
new_cmap = mcolors.LinearSegmentedColormap.from_list('hot_mod',
                    hot_cmap(np.linspace(0, 1, 256)**0.7))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
# print("Device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Torch version:", torch.__version__)


#%% Set up optical system
spacer=0

Num_pixel       = 180
Num_pupil1      = 150
FL              = 30e-2
wavelength      = 532e-9
pixel_size      = 8e-6
theta_R         = 1.22 * wavelength / (Num_pupil1*pixel_size)
x_R             = FL * theta_R
Rayleighlimit   = x_R / pixel_size

#For plotting images zoomed in on relevant section
range_img       = slice(Num_pixel//2 - 50, Num_pixel//2 + 50)
bias            = Num_pixel//2 - 50

#Scene parameters for test image
Sep_pixel       = 0.7*Rayleighlimit
b_list = [1, 2, 3, 4, 5]
n_phot = 5e4
phot_norm = n_phot
rel_b = b_list/np.sum(b_list)
n_norm = rel_b * n_phot / phot_norm


x_list = [-Sep_pixel/np.sqrt(2), -Sep_pixel/np.sqrt(2), Sep_pixel/np.sqrt(2), Sep_pixel/np.sqrt(2), 0.0]
y_list = [-Sep_pixel/np.sqrt(2), Sep_pixel/np.sqrt(2), -Sep_pixel/np.sqrt(2), Sep_pixel/np.sqrt(2), 0.0]
params_GT = torch.stack((torch.tensor(y_list, device=device), torch.tensor(x_list, device=device),
                           torch.tensor(n_norm, device=device)), dim=0)
Nsrc_fixed = (params_GT).shape[1]

#Create optical model object based on settings above
aper_size = (Num_pupil1)*pixel_size
model = AiryModel(Num_pixel, aper_size, pixel_size, wavelength, FL, device=device)

# Generate example image (no noise) and plot
Z_DI      = model.forward_model(params_GT, phot_norm)
Z_DINoise = torch.poisson(Z_DI)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
im1 = ax1.imshow(Z_DI[:, :].cpu().numpy())
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im1, cax=cax1)
ax1.set_xticks([])
ax1.set_yticks([])

im2 = ax2.imshow(Z_DINoise[range_img,range_img].cpu().numpy())
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im2, cax=cax2)
ax2.scatter(params_GT[1].cpu()+Num_pixel//2-bias, params_GT[0].cpu()+Num_pixel//2-bias, s=100, facecolors='none', edgecolors='orangered',
                                    linewidths=2, marker='o', label='True')
ax2.set_title(f'Separation: {round(Sep_pixel/Rayleighlimit, 1)}R')
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()

#%% Specify scene and calculate CRBs, QCRBs
spacer=0
num_config = 50
Nsrc_fixed = 5

#Specify parameters of the target scene
n_phot_lim = [5e5, 5e5]
sep_lim = [0.09*Rayleighlimit, 3.0*Rayleighlimit]
b_list = [1, 2, 3, 4, 5]
n_phot_list = np.logspace(np.log10(n_phot_lim[0]), np.log10(n_phot_lim[1]), num_config)
sep_list = np.logspace(np.log10(sep_lim[0]), np.log10(sep_lim[1]), num_config)

param_labels = ["Photon Number", "Separation (R)", "Relative Brightness"]
param_abbr = ["Photon Num", "Sep (R)", "Src2 RelBri", "Src3 RelBri"]
scan_param = 1          #0 phot num, 1 separation, 2 relative brightness

if scan_param == 0:
    plot_coord = n_phot_list
if scan_param == 1:
    plot_coord = sep_list/Rayleighlimit


param_labels = ["Photon Number", "Separation (R)"]

if scan_param == 0:
    plot_coord = n_phot_list
if scan_param == 1:
    plot_coord = sep_list/Rayleighlimit
rel_b = b_list/np.sum(b_list)

plot_lim = n_phot_lim
if scan_param == 1:
    plot_lim = sep_lim

## Caclulate Cramer Rao Lower Bounds
param_var_bnds_test = np.zeros((3*Nsrc_fixed, len(plot_coord)))
param_var_bnds_Q = np.zeros((3*Nsrc_fixed, len(plot_coord)))

for c_ind in range(num_config):
    n_phot_curr = n_phot_list[c_ind]
    sep_curr = sep_list[c_ind]

    phot_norm = n_phot_curr
    n_norm = rel_b*n_phot_curr/phot_norm

    #-- 5 source 'X' configuration
    x_list = [-sep_curr / np.sqrt(2), -sep_curr / np.sqrt(2), sep_curr / np.sqrt(2), sep_curr / np.sqrt(2), 0.0]
    y_list = [-sep_curr / np.sqrt(2), sep_curr / np.sqrt(2), -sep_curr / np.sqrt(2), sep_curr / np.sqrt(2), 0.0]
    params_curr = torch.stack((torch.tensor(y_list, device=device), torch.tensor(x_list, device=device),
                             torch.tensor(n_norm, device=device)), dim=0)

    J_mat = model.FIMcalc_analytic(params_curr, phot_norm)
    CRB_mat = torch.linalg.inv(J_mat).cpu().numpy()
    param_var_bnds_test[:, c_ind] = np.diag(CRB_mat)

    J_mat_Q = model.QFIMcalc(params_curr, phot_norm, method='SVD')
    QCRB_mat = torch.linalg.inv(J_mat_Q).cpu().numpy()
    param_var_bnds_Q[:, c_ind] = np.diag(QCRB_mat)

CRLB_y = param_var_bnds_test[0::3, :]
CRLB_x = param_var_bnds_test[1::3, :]
CRLB_b = (n_phot_list**2)*param_var_bnds_test[2::3, :]

QCRLB_y = param_var_bnds_Q[0::3, :]
QCRLB_x = param_var_bnds_Q[1::3, :]
QCRLB_b = param_var_bnds_Q[2::3, :]

CRLBMean_c = np.mean(np.sqrt(CRLB_x + CRLB_y), axis=0)
CRLBMean_b = np.mean(np.sqrt(CRLB_b), axis=0)
param_labels = ["Photon Number", "Separation (R)", "Relative Brightness"]
param_abbr = ["Photon Num", "Sep (R)", "Src2 RelBri"]



#%% Plot CRLB

CRLB_pos = CRLB_x + CRLB_y
CRLB_pos_mean = np.mean(CRLB_pos, axis=0)
CRLB_b_mean = np.mean(CRLB_b, axis=0)
QCRLB_pos = QCRLB_x + QCRLB_y
QCRLB_pos_mean = np.mean(QCRLB_pos, axis=0)
QCRLB_b_mean = np.mean(QCRLB_b, axis=0)

ymax_pos = 0.05
ymax_bri = 70000
# ymax_pos = None
# ymax_bri = None
ymin_pos = 0
ymin_bri = 0
# ymin_pos = 0.9*np.min(np.sqrt(QCRLB_pos_mean)/Rayleighlimit)
# ymin_bri = 0.9*np.min(np.sqrt(QCRLB_b_mean))

color_list = ['darkseagreen']
# fill_color = 'tab:orange'
fill_color = 'yellow'
#Plotting (all params)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
linewidth1 = 2.5
axs[0].axvline(1.0, color='black', linestyle='--', linewidth=linewidth1)
axs[1].axvline(1.0, color='black', linestyle='--', linewidth=linewidth1)


# Position CRLB
axs[0].plot(plot_coord, np.sqrt(CRLB_pos_mean)/Rayleighlimit, '-', color=color_list[0], linewidth=linewidth1, label=r"$\sqrt{\mathrm{Mean\;CRB}}$")
axs[0].plot(plot_coord, np.sqrt(QCRLB_pos_mean)/Rayleighlimit, '-', color='black', linewidth=linewidth1, label=r"$\sqrt{\mathrm{Mean\;QCRB}}$")
axs[0].fill_between(plot_coord, np.sqrt(QCRLB_pos_mean)/Rayleighlimit, np.sqrt(CRLB_pos_mean)/Rayleighlimit, color=fill_color, alpha=0.3)
axs[0].fill_between(plot_coord, 0, np.sqrt(QCRLB_pos_mean)/Rayleighlimit, color='grey', alpha=0.3)

# Brightness Subplot
# axs[1].plot(plot_coord, np.sqrt(CRLB_b_mean), '-', color=color_list[0], linewidth=linewidth1, label=r"$\sqrt{\mathrm{Mean\;CRB}}$")
# axs[1].plot(plot_coord, np.sqrt(QCRLB_b_mean), '-', color='black', linewidth=linewidth1, label=r"$\sqrt{\mathrm{Mean\;QCRB}}$")
axs[1].plot(plot_coord, np.sqrt(CRLB_b_mean), '-', color=color_list[0], linewidth=linewidth1, label="Standard\nImaging")
axs[1].plot(plot_coord, np.sqrt(QCRLB_b_mean), '-', color='black', linewidth=linewidth1, label="Quantum\nLimit")
axs[1].fill_between(plot_coord, np.sqrt(QCRLB_b_mean), np.sqrt(CRLB_b_mean), color=fill_color, alpha=0.3)
axs[1].fill_between(plot_coord, 0, np.sqrt(QCRLB_b_mean), color='grey', alpha=0.3)

axs[0].set_title("Position Error")
axs[1].set_title("Brightness Error")
# axs[0].set_xlim(0, sep_lim[1]/Rayleighlimit)
# axs[1].set_xlim(0, sep_lim[1]/Rayleighlimit)
axs[0].set_xlim(sep_lim[0]/Rayleighlimit, sep_lim[1]/Rayleighlimit)
axs[1].set_xlim(sep_lim[0]/Rayleighlimit, sep_lim[1]/Rayleighlimit)
# axs[0].set_xlim(0.0, 1.5)
# axs[1].set_xlim(0.0, 1.5)
axs[0].set_ylim(ymin_pos, ymax_pos)
axs[1].set_ylim(ymin_bri, ymax_bri)

# axs[0].set_xlim(np.min(plot_coord), None)

axs[0].set_xlabel(param_labels[scan_param])
axs[1].set_xlabel(param_labels[scan_param])
# axs[1].legend(loc='center right', fontsize=10)

# axs[0].set_ylabel(r"RMSE/Rayleigh Limit")
# axs[1].set_ylabel(r"RMSE (Photons)")
axs[0].set_ylabel(r"$\sqrt{\mathrm{Mean\;CRB}}$ (R)")
axs[1].set_ylabel(r"$\sqrt{\mathrm{Mean\;CRB}}$ (Photons)")

axs[0].set_xscale("log")
axs[1].set_xscale("log")
# axs[0].set_yscale("log")
# axs[1].set_yscale("log")

fig.tight_layout()
plt.show()


spacer=0








