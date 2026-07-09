#%% imports and settings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import math
from tqdm.auto import tqdm
import time
import pickle
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from UserPackage_AiryDiskModel import AiryModel, initialize_params_guess, Params_visualization, Reorder_estimate
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


#%% Set up optical system and show test image
Num_pixel       = 180
Num_pupil1      = 150       #Set aperture size of imaging system in pixels
wavelength      = 532e-9
pixel_size      = 8e-6
FL              = 30e-2
theta_R         = 1.22 * wavelength / (Num_pupil1*pixel_size)
x_R             = FL * theta_R
Rayleighlimit   = x_R / pixel_size          #Rayleigh resolution limit in units of pixels

Sep_pixel       = 1.5*Rayleighlimit         #Set separation for test image

range_img       = slice(Num_pixel//2 - 50, Num_pixel//2 + 50)       #Set crop region for viewing images
bias            = Num_pixel//2 - 50

#Parameters for test image
params_GT = torch.tensor([
    [0, -Sep_pixel/2, 1/3],
    [0, Sep_pixel/2, 2/3]], dtype=torch.float32, device= device).T
Nsrc_fixed = (params_GT).shape[1]

phot_norm = 1e6
aper_size = (Num_pupil1)*pixel_size
model = AiryModel(Num_pixel, aper_size, pixel_size, wavelength, FL, device=device)

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

spacer = 0
#%% Define scene parameters
spacer=0
num_config = 40
Nsrc_fixed = 3

# n_phot_lim = [5e3, 5e3]
# sep_lim = [0.2*Rayleighlimit, 1.5*Rayleighlimit]        #Specify separation values to test
# #Assume src1 has brightness 1
# b1_lim = [2, 2]
# b2_lim = [3, 3]

n_phot_lim = [5e4, 5e5]
sep_lim = [0.6*Rayleighlimit, 0.6*Rayleighlimit]        #Specify separation values to test
#Assume src1 has brightness 1
b1_lim = [2, 2]
b2_lim = [3, 3]

n_phot_list = np.logspace(np.log10(n_phot_lim[0]), np.log10(n_phot_lim[1]), num_config)
# sep_list = np.linspace(sep_lim[0], sep_lim[1], num_config)
sep_list = np.logspace(np.log10(sep_lim[0]), np.log10(sep_lim[1]), num_config)
b1_list = np.logspace(np.log10(b1_lim[0]), np.log10(b1_lim[1]), num_config)
b2_list = np.logspace(np.log10(b2_lim[0]), np.log10(b2_lim[1]), num_config)

param_labels = ["Photon Number", "Separation (R)", "Relative Brightness"]
param_abbr = ["Photon Num", "Sep (R)", "Src2 RelBri", "Src3 RelBri"]

#Specify which parameter is being scanned for plotting purposes. Set up to assume that all other parameters have same value
# for all configurations (i.e. set upper/lower limit to be equal above)
scan_param = 0          #0 phot num, 1 separation, 2 relative brightness

if scan_param == 0:
    plot_coord = n_phot_list
if scan_param == 1:
    plot_coord = sep_list/Rayleighlimit
if scan_param == 2:
    plot_coord = b1_list
if scan_param == 3:
    plot_coord = b2_list

#%% CRLB Testing - Calculate CRLB

param_var_bnds_test = np.zeros((3*Nsrc_fixed, len(plot_coord)))
for c_ind in range(num_config):
    n_phot_curr = n_phot_list[c_ind]
    sep_curr = sep_list[c_ind]
    b1_curr = b1_list[c_ind]
    b2_curr = b2_list[c_ind]

    phot_norm = n_phot_curr

    b0 = 1/(1+b1_curr+b2_curr)
    b1 = b1_curr/(1+b1_curr+b2_curr)
    b2 = b2_curr/(1+b1_curr+b2_curr)
    n0_norm = b0*n_phot_curr/phot_norm
    n1_norm = b1*n_phot_curr/phot_norm
    n2_norm = b2*n_phot_curr/phot_norm
    params_curr = torch.tensor([
        [0.0, -sep_curr/2, n0_norm],
        [0.0, sep_curr/2, n1_norm],
        [np.sqrt(3)/2 * sep_curr, 0, n2_norm]], dtype=torch.float32, device= device).T

    J_mat = model.FIMcalc_analytic(params_curr, phot_norm)
    CRB_mat = torch.linalg.inv(J_mat).cpu().numpy()
    param_var_bnds_test[:, c_ind] = np.diag(CRB_mat)

CRLB_y_test = param_var_bnds_test[0::3, :]
CRLB_x_test = param_var_bnds_test[1::3, :]
CRLB_b_test = (n_phot_list**2)*param_var_bnds_test[2::3, :]

# Plot CRLB bounds
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for k in range(Nsrc_fixed):
    axs[0].plot(plot_coord, np.sqrt(CRLB_x_test[k, :])/Rayleighlimit, color=color_list[k], label="Source " + str(k + 1))
    axs[1].plot(plot_coord, np.sqrt(CRLB_y_test[k, :])/Rayleighlimit, color=color_list[k], label="Source " + str(k + 1))
    axs[2].plot(plot_coord, np.sqrt(CRLB_b_test)[k, :], label="Source " + str(k + 1))

axs[0].set_title("X position")
axs[1].set_title("Y position")
axs[2].set_title("Brightness")
axs[0].set_xlabel(param_labels[scan_param])
axs[1].set_xlabel(param_labels[scan_param])
axs[2].set_xlabel(param_labels[scan_param])
axs[0].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$/Rayleigh Limit")
axs[1].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$/Rayleigh Limit")
axs[2].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$ (Photons)")

axs[1].set_title("Y position")
axs[2].set_title("Brightness")
axs[0].legend()
axs[1].legend()
axs[2].legend()
# axs[0].set_xscale('log')
# axs[1].set_xscale('log')
# axs[2].set_xscale('log')

fig.tight_layout()
plt.show()


#%% Do Estimation!
Scene_iter              = 100     #2000
init_rad = 1.0*Rayleighlimit            #set radius of random initialization region
min_init_sep = 0.25*Rayleighlimit       # set minimum separation for random initialization (don't want multiple sources on top of each other)
num_init = 2                            #Number of random initializations per trial, more initializations gives better chance of finding true MLE
conv_pos_eps =  Rayleighlimit*1e-6      #NG
conv_bri_rel_eps = 1e-3                 #NG

## Fisher Scoring/natural gradient descent settings
step_size_NG = 1e-1
reg_eps = 1e-1
Elearning_rate          = 0.2
Max_Eiter               = 2000

def lr_lambda(current_step):
    return 0.8 ** (current_step / Max_Eiter)

n_phot_curr = n_phot_lim[0]
sep_curr = sep_lim[0]
b1_curr = b1_lim[0]
b2_curr = b2_lim[0]

plot_lim = n_phot_lim
if scan_param == 1:
    plot_lim = sep_lim
if scan_param ==2:
    plot_lim = b1_lim
if scan_param ==3:
    plot_lim = b2_lim


#------ Specify paramter values to test - sample more values for harder parameter values, as
# ----- estimation has higher variance for harder scenes

# Linear sampling
linear_part = torch.empty(Scene_iter // 2).uniform_(plot_lim[0], plot_lim[1])
# Log sampling
log_min = math.log10(plot_lim[0])
log_max = math.log10(plot_lim[1])
log_part = 10 ** torch.empty(2*Scene_iter // 4).uniform_(log_min, log_max)
# log_part = 10 ** torch.empty(Scene_iter).uniform_(log_min, log_max)
# Combine
plot_input_arr = torch.cat([linear_part, log_part]).float()
# plot_input_arr = torch.cat([log_part]).float()
plot_input_arr = plot_input_arr.to(device)

# rel_bri_input_arr = torch.tensor([0.00001]).to(device)

model = model.to(device)

# Z_Noise_arr       = [None] * Scene_iter
Best_Parmas_arr   = [None] * Scene_iter

loop = tqdm(total=len(plot_input_arr),
            desc="Parameter Loop",
            dynamic_ncols=True,
            ascii=True,  # optional: visually simpler
            leave=False)  # 마지막에 깔끔히 지움

loss_arr = []
all_ests_arr = []
num_iter_arr = []
iter_time_arr = []; iter_time_list = []; t0 = time.time()
##-------- MONTE CARLO LOOP
for c_ind in range(Scene_iter):
    if scan_param == 0:
        n_phot_curr = plot_input_arr[c_ind]
        phot_norm = n_phot_curr
    if scan_param == 1:
        sep_curr = plot_input_arr[c_ind].cpu().numpy()
    if scan_param == 2:
        b1_curr = plot_input_arr[c_ind]
    if scan_param == 3:
        b2_curr = plot_input_arr[c_ind]

    b0 = 1/(1+b1_curr+b2_curr)
    b1 = b1_curr/(1+b1_curr+b2_curr)
    b2 = b2_curr/(1+b1_curr+b2_curr)
    n0_norm = b0*n_phot_curr/phot_norm
    n1_norm = b1*n_phot_curr/phot_norm
    n2_norm = b2*n_phot_curr/phot_norm
    params_curr = torch.tensor([
        [0.0, -sep_curr/2, n0_norm],
        [0.0, sep_curr/2, n1_norm],
        [np.sqrt(3)/2 * sep_curr, 0, n2_norm]], dtype=torch.float32, device= device).T

    Z_temp        = model.forward_model(params_curr, phot_norm)
    Z_Noise_temp         = torch.poisson(Z_temp)

    # # Initial guess equal brightness sources with half of total in photons in each
    # # Initial centroid guess random in region in center
    meas_phot = torch.sum(Z_Noise_temp)
    meas_phot_norm = meas_phot
    bri_guess = meas_phot / Nsrc_fixed

    params_np_list = []
    loss_list = []; iter_time_list = []; num_iter_list = [];
    for k in range(num_init):
        x_guesses = np.zeros(Nsrc_fixed)
        y_guesses = np.zeros(Nsrc_fixed)
        while np.sqrt((x_guesses[0] - x_guesses[1]) ** 2 + (y_guesses[0] - y_guesses[1]) ** 2) < min_init_sep:
            x_guesses = np.random.uniform(-init_rad, init_rad, Nsrc_fixed)
            y_guesses = np.random.uniform(-init_rad, init_rad, Nsrc_fixed)

        params_init = torch.tensor([
            [y_guesses[0], x_guesses[0], torch.log(bri_guess / meas_phot_norm)],
            [y_guesses[1], x_guesses[1], torch.log(bri_guess / meas_phot_norm)],
            [y_guesses[2], x_guesses[2], torch.log(bri_guess / meas_phot_norm)]], dtype=torch.float32, device=device).T
        params_guess = params_init.clone().detach()

        #### 1st EStep ####
        Eoptimizer = torch.optim.Adam([params_guess], lr=Elearning_rate)   # Optimizer definition
        Escheduler = torch.optim.lr_scheduler.LambdaLR(Eoptimizer, lr_lambda)

        ####-----------

        Estep = 0
        converged_var = False
        start = time.time()
        while not converged_var and Estep<Max_Eiter:
            Eoptimizer.zero_grad()
            params_old = params_guess.clone().detach()

            ### FISHER SCORING
            FIM = model.single_estep_analytic_NG(
                params_guess= params_guess,
                Z_noisy_arr=Z_Noise_temp,
                phot_norm=meas_phot_norm)
            FIM = FIM + reg_eps*torch.eye(Nsrc_fixed*3).to(device)
            FIM_inv = torch.linalg.inv(FIM)
            params_next = params_guess.T.flatten() - step_size_NG*torch.matmul(FIM_inv.to(torch.float32), params_guess.grad.T.flatten())
            params_next = torch.reshape(params_next, (Nsrc_fixed, 3)).T
            params_guess = params_next
            ### --------

            delta = (params_guess - params_old)
            delta_pos = torch.abs(delta[0:2, :])
            delta_bri = torch.abs(torch.exp(delta[2, :]) - 1)

            if torch.sum(delta_pos > conv_pos_eps)+torch.sum(delta_bri>conv_bri_rel_eps) == 0:
                converged_var = True


            Estep+=1
        end = time.time()

        num_iter_list.append(Estep)
        curr_loss = model.get_loss_analytic(params_guess, Z_Noise_temp, meas_phot_norm)
        if torch.sum(torch.isnan(params_guess)>0):
            curr_loss = torch.tensor(1.0, device=device)
        torch.cuda.empty_cache()

        params_guessAbs = params_guess.clone().detach()
        params_guessAbs[2] = torch.exp(params_guessAbs[2])*meas_phot_norm
        params_np_list.append(params_guessAbs.cpu().numpy())
        loss_list.append(curr_loss.detach().cpu().numpy())
        iter_time_list.append(end-start)

    #Pick best of the estimates from different initializations
    loss_arr.append(loss_list)
    all_ests_arr.append(params_np_list)
    num_iter_arr.append(num_iter_list)
    iter_time_arr.append(iter_time_list)

    est_ind = np.argmin(loss_list)
    params_np = params_np_list[est_ind]
    Best_Parmas_arr[c_ind]  = params_np
    end = time.time()
    loop.update(1)
loop.close()
t1 = time.time()


today = datetime.now()
date_str = today.strftime("D%m%d%y")  # D + MMDDYY
filename = (f"{date_str}_Phot{n_phot_lim[0]:.1e}to{n_phot_lim[1]:.1e}_Sep{sep_lim[0]/Rayleighlimit:.1e}to{sep_lim[1]/Rayleighlimit:.1e}R_"
            f"{date_str}_b1{b1_lim[0]:.1e}to{b1_lim[1]:.1e}_{Nsrc_fixed}src_{Scene_iter}Trials_NG.pkl")

save_dict = {
    "n_phot_lim": n_phot_lim, "sep_lim": sep_lim, "b1_lim": b1_lim, "b2_lim": b2_lim, "scan_param": scan_param,
    "Best_Parmas_arr": Best_Parmas_arr, "Nsrc_fixed": Nsrc_fixed,
    "plot_input_arr":  plot_input_arr, "Scene_iter": Scene_iter,
    "init_rad": init_rad, "min_init_sep": min_init_sep, "num_init": num_init,
    "Num_pixel": Num_pixel, "Num_pupil1": Num_pupil1, "wavelength": wavelength, "pixel_size": pixel_size, "FL": FL,
    "Elearning_rate": Elearning_rate, "Max_Eiter": Max_Eiter,
    "step_size_NG": step_size_NG, "reg_eps": reg_eps, "conv_pos_eps": conv_pos_eps, "conv_bri_rel_eps": conv_bri_rel_eps,
    "iter_time_arr": iter_time_arr, "num_iter_arr": num_iter_arr,
}

with open(filename, "wb") as f:
    pickle.dump(save_dict, f)

print("Done")
plot_input_arr = plot_input_arr.cpu().numpy()


#%%  %%%%%%%%%%%%%%% Plotting - Load data if desired and CRB Calculation %%%%%%%%%%%%%%%%
spacer=0
## ---- Load data if needed
# filename = (f"D022526_Phot5.0e+04to5.0e+04_Sep2.0e-01to1.5e+00R_D022526_b12.0e+00to2.0e+00_3src_2000Trials_NG.pkl")
#
# with open(filename, "rb") as f:
#     data_with = pickle.load(f)
# Best_Parmas_arr      = data_with['Best_Parmas_arr']
# plot_input_arr         = data_with['plot_input_arr']
# n_phot_lim = data_with['n_phot_lim']
# sep_lim = data_with['sep_lim']
# b1_lim = data_with['b1_lim']
# b2_lim = data_with['b2_lim']
# Nsrc_fixed = data_with['Nsrc_fixed']
# Scene_iter = data_with['Scene_iter']
# scan_param = data_with['scan_param']
# num_iter_arr = data_with['num_iter_arr']
# Max_Eiter = data_with['Max_Eiter']
# plot_input_arr = plot_input_arr.cpu().numpy()
### ------------

#Calculate CRBs
num_config = 40
n_phot_list = np.logspace(np.log10(n_phot_lim[0]), np.log10(n_phot_lim[1]), num_config)
sep_list = np.linspace(sep_lim[0], sep_lim[1], num_config)
b1_list = np.logspace(np.log10(b1_lim[0]), np.log10(b1_lim[1]), num_config)
b2_list = np.logspace(np.log10(b2_lim[0]), np.log10(b2_lim[1]), num_config)
param_labels = ["Photon Number", "Separation (R)", "Relative Brightness"]

if scan_param == 0:
    plot_coord = n_phot_list
if scan_param == 1:
    plot_coord = sep_list/Rayleighlimit
if scan_param == 2:
    plot_coord = b1_list
if scan_param == 3:
    plot_coord = b2_list

plot_lim = n_phot_lim
if scan_param == 1:
    plot_lim = sep_lim
if scan_param ==2:
    plot_lim = b1_lim
if scan_param ==3:
    plot_lim = b2_lim

## Check CRLB for these relative brightnesses
param_var_bnds_test = np.zeros((3*Nsrc_fixed, len(plot_coord)))
for c_ind in range(num_config):
    n_phot_curr = n_phot_list[c_ind]
    sep_curr = sep_list[c_ind]
    b1_curr = b1_list[c_ind]
    b2_curr = b2_list[c_ind]

    phot_norm = n_phot_curr

    b0 = 1/(1+b1_curr+b2_curr)
    b1 = b1_curr/(1+b1_curr+b2_curr)
    b2 = b2_curr/(1+b1_curr+b2_curr)
    n0_norm = b0*n_phot_curr/phot_norm
    n1_norm = b1*n_phot_curr/phot_norm
    n2_norm = b2*n_phot_curr/phot_norm
    params_curr = torch.tensor([
        [0.0, -sep_curr/2, n0_norm],
        [0.0, sep_curr/2, n1_norm],
        [np.sqrt(3)/2*sep_curr, 0, n2_norm]], dtype=torch.float32, device= device).T

    J_mat = model.FIMcalc_analytic(params_curr, phot_norm)
    CRB_mat = torch.linalg.inv(J_mat).cpu().numpy()
    param_var_bnds_test[:, c_ind] = np.diag(CRB_mat)

CRLB_y = param_var_bnds_test[0::3, :]
CRLB_x = param_var_bnds_test[1::3, :]
CRLB_b = (n_phot_list**2)*param_var_bnds_test[2::3, :]

CRLBMean_c = np.mean(np.sqrt(CRLB_x + CRLB_y), axis=0)
CRLBMean_b = np.mean(np.sqrt(CRLB_b), axis=0)
param_labels = ["Photon Number", "Separation (R)", "Relative Brightness"]
param_abbr = ["Photon Num", "Sep (R)", "Src2 RelBri"]

#%% Assign point sources
est_x_ord = np.zeros((Nsrc_fixed, Scene_iter))
est_y_ord = np.zeros((Nsrc_fixed, Scene_iter))
est_bri_ord = np.zeros((Nsrc_fixed, Scene_iter))

#In this section, we match the estimated point sources to the corresponding ground truth point sources by
# finding the assignment that minimizes error

beta = None       #0 for assignment based only on position, large for brightness based, None for autoscale

n_phot_curr = n_phot_lim[0]
sep_curr = sep_lim[0]
b1_curr = b1_lim[0]
b2_curr = b2_lim[0]
for c_ind in range(Scene_iter):
    if scan_param == 0:
        n_phot_curr = plot_input_arr[c_ind]
        phot_norm = n_phot_curr
    if scan_param == 1:
        sep_curr = plot_input_arr[c_ind]
    if scan_param == 2:
        b1_curr = plot_input_arr[c_ind]
    if scan_param == 3:
        b2_curr = plot_input_arr[c_ind]

    b0 = 1/(1+b1_curr+b2_curr)
    b1 = b1_curr/(1+b1_curr+b2_curr)
    b2 = b2_curr/(1+b1_curr+b2_curr)
    n0_norm = b0*n_phot_curr/phot_norm
    n1_norm = b1*n_phot_curr/phot_norm
    n2_norm = b2*n_phot_curr/phot_norm
    params_GT = torch.tensor([
        [0.0, -sep_curr/2, n0_norm],
        [0.0, sep_curr/2, n1_norm],
        [np.sqrt(3)/2*sep_curr, 0, n2_norm]], dtype=torch.float32, device= device).T

    params_est = np.copy(Best_Parmas_arr[c_ind])
    params_est[2, :] = params_est[2, :]/phot_norm

    _, _, Matching_Ind = Params_visualization(params_GT, params_est, Num_pixel, bias, beta=beta)
    params_reordered = Reorder_estimate(params_est, Matching_Ind)

    est_x_ord[:, c_ind] = params_reordered[1, :]
    est_y_ord[:, c_ind] = params_reordered[0, :]
    est_bri_ord[:, c_ind] = params_reordered[2, :]*phot_norm


#%% Calculate RMSEs of estimates
spacer = 0
num_bins = 12
# num_bins = 100
# bins = np.linspace(plot_lim[0], plot_lim[1], num_bins + 1)
bins = np.logspace(np.log10(plot_lim[0]), np.log10(plot_lim[1]), num_bins + 1)

# 2. 평균 계산
bin_centers = []
bin_mean_cerrs = []
bin_bmeans = []
bin_x_MSE = []
bin_y_MSE = []
bin_bri_MSE = []

y_gt_arr = np.ones((Scene_iter, Nsrc_fixed)) * [0.0, 0.0, np.sqrt(3)/2*sep_lim[0]]
x_gt_arr = np.ones((Scene_iter, Nsrc_fixed)) * [-sep_lim[0]/2, sep_lim[0]/2, 0]
b0 = (1/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b1 = (b1_lim[0]/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b2 = (b2_lim[0]/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b_gt_arr = np.ones((Scene_iter, Nsrc_fixed)) * [b0, b1, b2]

if scan_param == 0:
    b0_temp = (1 / (1 + b1_lim[0] + b2_lim[0])) * plot_input_arr
    b1_temp = (b1_lim[0] / (1 + b1_lim[0] + b2_lim[0])) * plot_input_arr
    b2_temp = (b2_lim[0] / (1 + b1_lim[0] + b2_lim[0])) * plot_input_arr
    b_gt_arr = np.stack([b0_temp, b1_temp, b2_temp]).T
if scan_param == 1:
    x_gt_arr = np.stack([-plot_input_arr/2, plot_input_arr/2, np.zeros(Scene_iter)]).T
    y_gt_arr = np.stack([np.zeros(Scene_iter), np.zeros(Scene_iter), np.sqrt(3)/2*plot_input_arr]).T
if scan_param == 2:
    b0_temp = 1/(1 + plot_input_arr + b2_lim[0]) * n_phot_lim[0]
    b1_temp = plot_input_arr/(1 + plot_input_arr + b2_lim[0]) * n_phot_lim[0]
    b2_temp = b2_lim[0]/(1 + plot_input_arr + b2_lim[0]) * n_phot_lim[0]
    b_gt_arr = np.stack([b0_temp, b1_temp, b2_temp]).T

ests_arr_np = np.array(Best_Parmas_arr)
x_errs = np.swapaxes(est_x_ord, 0, 1) - x_gt_arr
y_errs = np.swapaxes(est_y_ord, 0, 1) - y_gt_arr
bri_errs = np.swapaxes(est_bri_ord, 0, 1) - b_gt_arr

for i in range(num_bins):
    mask = ((plot_input_arr >= bins[i]) & (plot_input_arr < bins[i+1]))

    # See what happens if we filter out some outliers
    # out_mask = np.abs(bri_errs[:, 0])>1e6
    # mask = mask & (~out_mask)

    if np.any(mask):
        if scan_param == 1:
            bin_centers.append(((bins[i] + bins[i+1]) / 2)/Rayleighlimit)
        else:
            bin_centers.append((bins[i] + bins[i+1]) / 2)

        bin_x_MSE.append(np.mean((x_errs[mask])**2, axis=0))
        bin_y_MSE.append(np.mean((y_errs[mask])**2, axis=0))
        bin_bri_MSE.append(np.mean((bri_errs[mask])**2, axis=0))

bin_RMSE_x = np.sqrt(bin_x_MSE)
bin_RMSE_y = np.sqrt(bin_y_MSE)
bin_RMSE_b = np.sqrt(bin_bri_MSE)
# bin_RMSE_b = np.sqrt(bin_rbri_MSE)

bin_meanRMSE_x = np.mean(np.sqrt(bin_x_MSE), axis=1)
bin_meanRMSE_y = np.mean(np.sqrt(bin_y_MSE), axis=1)
bin_meanRMSE_c = np.mean(np.sqrt(np.array(bin_x_MSE) + np.array(bin_y_MSE)), axis=1)
bin_meanRMSE_b = np.mean(np.sqrt(bin_bri_MSE), axis=1)

#%% Plot individual estimates
y_plot_arr = np.ones((num_config, Nsrc_fixed)) * [0.0, 0.0, np.sqrt(3)/2*sep_lim[0]]/Rayleighlimit
x_plot_arr = np.ones((num_config, Nsrc_fixed)) * [-sep_lim[0]/2, sep_lim[0]/2, 0]/Rayleighlimit
b0 = (1/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b1 = (b1_lim[0]/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b2 = (b2_lim[0]/(1+b1_lim[0] + b2_lim[0]))*n_phot_lim[0]
b_plot_arr = np.ones((num_config, Nsrc_fixed)) * [b0, b1, b2]
scatt_arr = plot_input_arr

if scan_param == 0:
    b0_temp = (1 / (1 + b1_lim[0] + b2_lim[0])) * plot_coord
    b1_temp = (b1_lim[0] / (1 + b1_lim[0] + b2_lim[0])) * plot_coord
    b2_temp = (b2_lim[0] / (1 + b1_lim[0] + b2_lim[0])) * plot_coord
    b_plot_arr = np.stack([b0_temp, b1_temp, b2_temp]).T
if scan_param == 1:
    x_plot_arr = np.stack([-plot_coord/2, plot_coord/2, np.zeros(num_config)]).T
    y_plot_arr = np.stack([np.zeros(num_config), np.zeros(num_config), np.sqrt(3)/2*plot_coord*Rayleighlimit]).T/Rayleighlimit
    scatt_arr = plot_input_arr/Rayleighlimit
if scan_param == 2:
    b0_temp = 1/(1 + plot_coord + b2_lim[0]) * n_phot_lim[0]
    b1_temp = plot_coord/(1 + plot_coord + b2_lim[0]) * n_phot_lim[0]
    b2_temp = b2_lim[0]/(1 + plot_coord + b2_lim[0]) * n_phot_lim[0]
    b_plot_arr = np.stack([b0_temp, b1_temp, b2_temp]).T


color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, axs = plt.subplots(1, 3, figsize = (10,4))
alpha_val = 0.1     #Scatter points opacity, adjust based on number of points


for k in range(Nsrc_fixed):
    axs[0].scatter(scatt_arr, est_x_ord[k, :]/Rayleighlimit,
                label=f'Source {k} Estimates', alpha=alpha_val, s=25, color=color_list[k])
    axs[0].plot(plot_coord, x_plot_arr[:, k],
                label=f'Source {k} GT', linestyle='--', color=color_list[k])

    axs[1].scatter(scatt_arr, est_y_ord[k, :]/Rayleighlimit,
                label=f'Source {k} Estimates', alpha=alpha_val, s=25, color=color_list[k])
    axs[1].plot(plot_coord, y_plot_arr[:, k],
                label=f'Source {k} GT', linestyle='--', color=color_list[k])

    axs[2].scatter(scatt_arr, est_bri_ord[k, :],
                label=f'Source {k} Estimates', alpha=alpha_val, s=25, color=color_list[k])
    axs[2].plot(plot_coord, b_plot_arr[:, k],
                label=f'Source {k} GT', linestyle='--', color=color_list[k])

axs[0].set_ylabel('X Estimates (R)')
axs[0].set_xlabel(param_labels[scan_param])
axs[1].set_ylabel('Y Estimates (R)')
axs[1].set_xlabel(param_labels[scan_param])
axs[2].set_ylabel('Brightness Estimates (Photons)')
axs[2].set_xlabel('Relative Brightness')
axs[2].set_ylabel('Brightness Estimates (Photons)')
axs[2].set_xlabel(param_labels[scan_param])
# axs[0].set_xscale('log')
# axs[1].set_xscale('log')
# axs[2].set_xscale('log')
# axs[2].set_yscale('log')

# axs[0].set_ylim([-0.3, 0.3])
# axs[1].set_ylim([-1, 1])
# axs[2].set_ylim([-1e5, 1.2e6])
# axs[1].legend()
# plt.legend()
plt.tight_layout()
plt.show()



#%% Compare to CRLB parameter by parameter

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#Plotting (all params)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

for k in range(Nsrc_fixed):
    axs[0].plot(plot_coord, np.sqrt(CRLB_x[k, :])/Rayleighlimit, '--', color=color_list[k], label="CRLB, Source " + str(k + 1))
    axs[0].plot(np.array(bin_centers), bin_RMSE_x[:, k]/Rayleighlimit, color=color_list[k], label="Est RMSE, Source " + str(k + 1))

    axs[1].plot(plot_coord, np.sqrt(CRLB_y[k, :])/Rayleighlimit, '--', color=color_list[k], label="CRLB, Source " + str(k + 1))
    axs[1].plot(np.array(bin_centers), bin_RMSE_y[:, k]/Rayleighlimit, color=color_list[k], label="Source " + str(k + 1))

    # axs[2].plot(plot_coord, np.sqrt(CRLB_b[k, :]), '--', color=color_list[k], label="CRLB, Source " + str(k + 1))
    # axs[2].plot(np.array(bin_centers), bin_RMSE_b[:, k], color=color_list[k], label= "Est RMSE, Source " + str(k + 1))
    axs[2].plot(plot_coord, np.sqrt(CRLB_b[k, :]), '--', color=color_list[k], label="CRLB, Source " + str(k + 1))
    axs[2].plot(np.array(bin_centers), bin_RMSE_b[:, k], color=color_list[k], label= "Est RMSE, Source " + str(k + 1))


axs[0].set_title("X position")
axs[1].set_title("Y position")
axs[2].set_title("Brightness")
axs[0].set_xlabel(param_labels[scan_param])
axs[1].set_xlabel(param_labels[scan_param])
axs[2].set_xlabel(param_labels[scan_param])
axs[0].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$/Rayleigh Limit")
axs[1].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$/Rayleigh Limit")
axs[2].set_ylabel(r"$\sqrt{\mathrm{CRLB}}$ (Photons)")

axs[1].set_title("Y position")
axs[2].set_title("Brightness")
axs[0].legend()
axs[1].legend()
axs[2].legend(loc='upper right')

fig.tight_layout()
plt.show()





