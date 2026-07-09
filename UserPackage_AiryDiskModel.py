import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn

class AiryModel(nn.Module):
    def __init__(self, Num_pixel, aper_size, pixel_size, wavelength, FL, device = 'cpu'):
        super(AiryModel, self).__init__()
        self.device = torch.device(device)
        self.Num_pixel = Num_pixel
        #  Parameters
        self.register_buffer("wavelength", torch.tensor(wavelength, dtype=torch.float64, device=self.device))
        self.register_buffer("pixel_size", torch.tensor(pixel_size, dtype=torch.float64, device=self.device))
        self.register_buffer("FL", torch.tensor(FL, dtype=torch.float64, device=self.device))
        self.register_buffer("k", torch.tensor(2 * np.pi / wavelength, dtype=torch.float64, device=self.device))
        self.register_buffer("D", torch.tensor(aper_size, dtype=torch.float64, device=self.device))
        norm_const = self.pixel_size ** 2 * np.pi * (self.D / 2) ** 2 / (self.wavelength * self.FL) ** 2
        self.register_buffer("Airy_norm", norm_const)
        self.register_buffer("RL", 1.22 * self.wavelength * self.FL / self.D)

        # Coordinate grid
        x = torch.linspace(-Num_pixel / 2, Num_pixel / 2 - 1, Num_pixel, device=self.device) * self.pixel_size
        y = x.clone()
        dx, dy = torch.meshgrid(x, y, indexing='xy')
        self.register_buffer("dx", dx.unsqueeze(0))  # [1, H, W]
        self.register_buffer("dy", dy.unsqueeze(0))  # [1, H, W]


        # Aperture
        pupil1 = self.AperConst(aper_size)
        pupil1_norm = pupil1 / torch.linalg.norm(pupil1)
        self.register_buffer("Const_pupil1", pupil1)
        self.register_buffer("Const_pupil1Norm", pupil1_norm)


        # Get normalization constant for

    def AperConst(self, aper_size):
        r = torch.sqrt(self.dx[0, :, :] ** 2 + self.dy[0, :, :] ** 2)
        mask = (r <= aper_size/2).float()
        return mask

    def forward_model(self, ParamsArray, phot_norm):
        x_shift = ParamsArray[1] * self.pixel_size
        y_shift = ParamsArray[0] * self.pixel_size
        brightness = ParamsArray[2] * phot_norm

        R = torch.sqrt((self.dx - x_shift[:, None, None]) ** 2 + (self.dy - y_shift[:, None, None]) ** 2)
        R = torch.where(R == 0, 1e-20 * torch.ones_like(R), R)
        arg = 1.22 * np.pi * R / self.RL

        Zarr = torch.sum(brightness[:, None, None] * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg) ** 2, axis=0)

        return Zarr

    def AiryGrad_analytic(self, ParamsArray, phot_norm):
        # all derivatives with respect to parameters defined in ParamsArray (so scaling meant to match definition from ParamsArray)
        x_shift = ParamsArray[1] * self.pixel_size
        y_shift = ParamsArray[0] * self.pixel_size
        brightness = ParamsArray[2] * phot_norm

        R = torch.sqrt((self.dx - x_shift[:, None, None]) ** 2 + (self.dy - y_shift[:, None, None]) ** 2)
        R = torch.where(R == 0, 1e-20 * torch.ones_like(R), R)
        arg = 1.22 * np.pi * R / self.RL

        dairy_db = phot_norm * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg) ** 2
        dairy_darg = 4 * brightness[:, None, None] * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg ** 2) * (
                    torch.special.bessel_j0(arg) - 2 * torch.special.bessel_j1(arg) / arg)
        darg_dx = -(1.22 * torch.pi / self.RL) * (self.dx - x_shift[:, None, None]) / R
        darg_dy = -(1.22 * torch.pi / self.RL) * (self.dy - y_shift[:, None, None]) / R

        dairy_dx = dairy_darg * darg_dx * self.pixel_size
        dairy_dy = dairy_darg * darg_dy * self.pixel_size

        return torch.permute(torch.stack((dairy_dy, dairy_dx, dairy_db)), (2, 3, 0, 1))

    def FIMcalc_analytic(self, ParamsArray, phot_norm):
        # all derivatives with respect to parameters defined in ParamsArray (so scaling meant to match definition from ParamsArray)
        x_shift = ParamsArray[1] * self.pixel_size
        y_shift = ParamsArray[0] * self.pixel_size
        brightness = ParamsArray[2] * phot_norm

        R = torch.sqrt((self.dx - x_shift[:, None, None]) ** 2 + (self.dy - y_shift[:, None, None]) ** 2)
        R = torch.where(R == 0, 1e-20 * torch.ones_like(R), R)
        arg = 1.22 * np.pi * R / self.RL

        dairy_db = phot_norm * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg) ** 2
        I_gt = torch.sum(brightness[:, None, None] * dairy_db / phot_norm, axis=(0))
        dairy_darg = 4 * brightness[:, None, None] * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg ** 2) * (
                    torch.special.bessel_j0(arg) - 2 * torch.special.bessel_j1(arg) / arg)
        darg_dx = -(1.22 * torch.pi / self.RL) * (self.dx - x_shift[:, None, None]) / R
        darg_dy = -(1.22 * torch.pi / self.RL) * (self.dy - y_shift[:, None, None]) / R

        dairy_dx = dairy_darg * darg_dx * self.pixel_size
        dairy_dy = dairy_darg * darg_dy * self.pixel_size

        grad_arr = torch.permute(
            torch.flatten(torch.permute(torch.stack((dairy_dy, dairy_dx, dairy_db)), (2, 3, 1, 0)), start_dim=2,
                          end_dim=3), (2, 0, 1))

        FIM = torch.einsum('kmi, jmi -> kj', grad_arr / torch.sqrt(I_gt), grad_arr / torch.sqrt(I_gt))
        return FIM

    def computeFIM_DI_auto(self, ParamsArray, grad_func, phot_norm):
        I_gt = self.forward_model(ParamsArray, phot_norm)
        grad_temp = grad_func(ParamsArray)
        grad_arr = torch.permute(torch.flatten(torch.transpose(grad_temp, 2, 3), start_dim=2, end_dim=3), (2, 0, 1))
        FIM = torch.einsum('kmi, jmi -> kj', grad_arr / torch.sqrt(I_gt), grad_arr / torch.sqrt(I_gt))
        return FIM

    def single_estep_analytic(self, params_guess, Z_noisy_arr, phot_norm):
        # all derivatives with respect to parameters defined in ParamsArray (so scaling meant to match definition from ParamsArray)
        #Just for gradient descent calculations, use LOG BRIGHTNESS!!!
        x_shift = params_guess[1] * self.pixel_size
        y_shift = params_guess[0] * self.pixel_size
        brightness = torch.exp(params_guess[2]) * phot_norm

        R = torch.sqrt((self.dx - x_shift[:, None, None]) ** 2 + (self.dy - y_shift[:, None, None]) ** 2)
        R = torch.where(R == 0, 1e-20 * torch.ones_like(R), R)
        arg = 1.22 * np.pi * R / self.RL

        airy_prof = self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg) ** 2
        dairy_db = torch.exp(params_guess[2][:, None, None]) * phot_norm * airy_prof
        I_gt = torch.sum(brightness[:, None, None] * airy_prof, axis=(0))
        dairy_darg = 4 * brightness[:, None, None] * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg ** 2) * (
                torch.special.bessel_j0(arg) - 2 * torch.special.bessel_j1(arg) / arg)
        darg_dx = -(1.22 * torch.pi / self.RL) * (self.dx - x_shift[:, None, None]) / R
        darg_dy = -(1.22 * torch.pi / self.RL) * (self.dy - y_shift[:, None, None]) / R

        dairy_dx = dairy_darg * darg_dx * self.pixel_size
        dairy_dy = dairy_darg * darg_dy * self.pixel_size

        mult_arr = (Z_noisy_arr/I_gt - 1)
        dloss_dx = -torch.sum(dairy_dx * mult_arr, axis=(1, 2))
        dloss_dy = -torch.sum(dairy_dy * mult_arr, axis=(1, 2))
        dloss_db = -torch.sum(dairy_db * mult_arr, axis=(1, 2))
        dloss_arr = torch.stack([dloss_dy, dloss_dx, dloss_db])
        params_guess.grad = dloss_arr

    def single_estep_analytic_NG(self, params_guess, Z_noisy_arr, phot_norm):
        # all derivatives with respect to parameters defined in ParamsArray (so scaling meant to match definition from ParamsArray)
        #Just for gradient descent calculations, use LOG BRIGHTNESS!!!
        x_shift = params_guess[1] * self.pixel_size
        y_shift = params_guess[0] * self.pixel_size
        brightness = torch.exp(params_guess[2]) * phot_norm

        R = torch.sqrt((self.dx - x_shift[:, None, None]) ** 2 + (self.dy - y_shift[:, None, None]) ** 2)
        R = torch.where(R == 0, 1e-20 * torch.ones_like(R), R)
        arg = 1.22 * np.pi * R / self.RL

        airy_prof = self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg) ** 2
        dairy_db = torch.exp(params_guess[2][:, None, None]) * phot_norm * airy_prof
        I_gt = torch.sum(brightness[:, None, None] * airy_prof, axis=(0))
        dairy_darg = 4 * brightness[:, None, None] * self.Airy_norm * (2 * torch.special.bessel_j1(arg) / arg ** 2) * (
                torch.special.bessel_j0(arg) - 2 * torch.special.bessel_j1(arg) / arg)
        darg_dx = -(1.22 * torch.pi / self.RL) * (self.dx - x_shift[:, None, None]) / R
        darg_dy = -(1.22 * torch.pi / self.RL) * (self.dy - y_shift[:, None, None]) / R

        dairy_dx = dairy_darg * darg_dx * self.pixel_size
        dairy_dy = dairy_darg * darg_dy * self.pixel_size

        mult_arr = (Z_noisy_arr/I_gt - 1)
        dloss_dx = -torch.sum(dairy_dx * mult_arr, axis=(1, 2))
        dloss_dy = -torch.sum(dairy_dy * mult_arr, axis=(1, 2))
        dloss_db = -torch.sum(dairy_db * mult_arr, axis=(1, 2))
        dloss_arr = torch.stack([dloss_dy, dloss_dx, dloss_db])
        params_guess.grad = dloss_arr

        grad_arr = torch.permute(torch.flatten(torch.permute(torch.stack((dairy_dy, dairy_dx, dairy_db)), (2, 3, 1, 0)), start_dim=2,
                          end_dim=3), (2, 0, 1))

        FIM = torch.einsum('kmi, jmi -> kj', grad_arr / torch.sqrt(I_gt), grad_arr / torch.sqrt(I_gt))
        return FIM

    def get_loss_analytic(self, params_guess, Z_noisy_arr, phot_norm):
        params_guess_temp = params_guess.clone()
        params_guess_temp[2] = torch.exp(params_guess_temp[2])
        Z_guess = self.forward_model(params_guess_temp, phot_norm)
        poisson_loss = torch.sum(-Z_noisy_arr * torch.log(Z_guess) + Z_guess)
        loss = poisson_loss
        return loss

    def pupil_fields(self, ParamsArray, phot_norm):
        epsilon = 1e-18  # float64에서는 더 작은 epsilon 사용 가능
        device = self.dx.device

        # 입력 텐서들을 명시적으로 float64/complex128로 변환 및 장치 동기화
        # ParamsArray: [3, 3] -> x_shift: [3, 1, 1]
        x_shift = (ParamsArray[1].to(device, dtype=torch.float64))[:, None, None]
        y_shift = (ParamsArray[0].to(device, dtype=torch.float64))[:, None, None]
        brightness = phot_norm*ParamsArray[2].to(device, dtype=torch.float64)[:, None, None]

        amplitude = torch.sqrt(torch.abs(brightness) + epsilon)

        # 1. Tip-Tilt Phase 생성 (complex128)
        phase_arg = -1j * self.k.to(torch.complex128) * (y_shift * self.dy + x_shift * self.dx) * self.pixel_size / self.FL
        # print(phase_arg)
        TipTiltPhase = amplitude.to(torch.complex128) * self.Const_pupil1Norm[None] * torch.exp(phase_arg)
        TipTiltPhase = TipTiltPhase.flatten(start_dim=1, end_dim=2)

        return torch.cat((TipTiltPhase.real, TipTiltPhase.imag), dim=1)

    def QFIMcalc(self, ParamsArray, phot_norm, method='SVD'):
        Nsrc_fixed = ParamsArray.shape[1]
        Num_pixel = self.Num_pixel
        device = self.device

        bn = ParamsArray[2, :] / torch.sum(ParamsArray[2, :])
        params_quantum = ParamsArray.clone().detach()
        params_quantum[2, :] = torch.ones(Nsrc_fixed) / phot_norm
        tot_phot = torch.sum(ParamsArray[2, :]) * phot_norm


        ######----- Calculate relevant E-fields and derivatives
        #Get pupil plane E-fields
        fields_func = lambda params: self.pupil_fields(params, phot_norm)
        grad_func = torch.func.jacfwd(fields_func)

        fields_aug = fields_func(params_quantum)
        fields = (fields_aug[:, 0:Num_pixel ** 2] + 1j * fields_aug[:,Num_pixel ** 2:]).T  # Put states as column vectors
        field_derivs_aug = grad_func(params_quantum)
        field_derivs_extra = field_derivs_aug[:, 0:Num_pixel ** 2, :, :] + 1j * field_derivs_aug[:, Num_pixel ** 2:, :,:]
        field_derivs_relevant = torch.diagonal(field_derivs_extra, dim1=0, dim2=3)[:, 0:3, :].transpose(dim0=1, dim1=2).flatten(start_dim=1, end_dim=2)
        # field_derivs_relevant [Num_pixel**2, num_param], y1, x1, bri1, y2,..briN ordering

        ##### ---- Construct orthonormal basis for density matrix and derivatives
        # All vectors needed are in field_derivs_relevant (bri derivs are same as field modes)
        V_basis = field_derivs_relevant  # Each linearly independent vector is column of V_basis, not orthogonal basis
        if method=='Gram':
            # Construct Gram matrix and use to form orthonormal basis
            G = V_basis.H @ V_basis
            Dval, Tg = torch.linalg.eigh(G)  # eignvals + eigenvecs of Gram matrix. Eignvals real and eigvenvecs orthonormal because G Hermitian
            Dg_sqrtInv = torch.diag(1 / torch.sqrt(Dval)).to(torch.complex128)
            U_basis = V_basis @ Tg @ Dg_sqrtInv  # U orthogonal basis with same span as V_basis
        if method=='SVD':
            #Use SVD to form orthonormal basis
            _, _, U_basis = torch.linalg.svd(V_basis.T, full_matrices=False)
            U_basis = U_basis.T

        # Represent field vectors in the constructed basis
        # ---check inner product of fields_U, seems off rn, may be due to sampling being small
        fields_U = U_basis.H @ fields
        # fields_check = U_basis@fields_U
        field_derivs_U = U_basis.H @ field_derivs_relevant  # represent field derivatives in the U basis for later

        ####---- Build density matrix in U basis representation
        rho = torch.zeros(3*Nsrc_fixed, 3*Nsrc_fixed, device=device, dtype=fields_U.dtype)
        for k in range(Nsrc_fixed):
            rho += torch.outer(bn[k] * (fields_U[:, k]), torch.conj(fields_U[:, k]))
            # rho += torch.outer(torch.conj(fields_U[k, :]), (fields_U[k, :]))

        # Get eigenvalues and eigenvectors of density matrix in U_basis
        lamb, E_basis = torch.linalg.eigh(rho)
        rho_E = torch.diag(lamb)
        # Write density matrix derivatives in eigenbasis
        field_derivs_E = E_basis.H @ field_derivs_U
        # field_derivs_check = U_basis@E_basis@field_derivs_E
        fields_E = E_basis.H @ fields_U

        ###### ------- Calculate SLD operator for each parameter

        # write derivatives of density matrices in E_basis
        drho_dtheta_arr = torch.zeros(3 * Nsrc_fixed, 3 * Nsrc_fixed, 3 * Nsrc_fixed, device=device,dtype=torch.complex128)  # [param_ind, basis ind, basis ind]
        for m in range(3 * Nsrc_fixed):
            if m % 3 == 2:
                src_ind = m // 3
                drho_dtheta_arr[m, :, :] = torch.outer((fields_E[:, src_ind]), torch.conj(fields_E[:, src_ind]))
            else:
                src_ind = m // 3
                drho_dtheta_arr[m, :, :] = bn[src_ind] * (torch.outer((field_derivs_E[:, m]), torch.conj(fields_E[:, src_ind])) + torch.outer((fields_E[:, src_ind]), torch.conj(field_derivs_E[:, m])))

        lamb_row, lamb_col = torch.meshgrid(lamb, lamb, indexing='ij')
        denominator = (lamb_row + lamb_col).to(torch.complex128)
        mask_ind = torch.abs(denominator) < 1e-15
        SLD_arr = 2 * drho_dtheta_arr / denominator
        SLD_arr[:, mask_ind] = 0

        QFIM_onephot = torch.zeros(3 * Nsrc_fixed, 3 * Nsrc_fixed, device=device, dtype=torch.float64)
        for l in range(3 * Nsrc_fixed):
            for m in range(3 * Nsrc_fixed):
                QFIM_onephot[l, m] = torch.trace(SLD_arr[l, :, :] @ SLD_arr[m, :, :] @ rho_E.to(torch.complex128)).real
                if l % 3 == 2:
                    QFIM_onephot[l, m] = QFIM_onephot[l, m] / tot_phot
                if m % 3 == 2:
                    QFIM_onephot[l, m] = QFIM_onephot[l, m] / tot_phot

        QFIM = tot_phot * QFIM_onephot
        return QFIM


def initialize_params_guess(N, Sep_pixel, device='cpu'):
    min_distance = 1.0
    max_attempts = 100

    for attempt in range(max_attempts):
        xy = (torch.rand(2, N, dtype=torch.float32, device=device) * Sep_pixel) - (Sep_pixel / 2)  # [2, N]

        xy_T = xy.t()  # [N, 2]
        diffs = xy_T.unsqueeze(1) - xy_T.unsqueeze(0)  # [N, N, 2]
        dists = torch.norm(diffs, dim=-1)  # [N, N]
        dists += torch.eye(N, device=dists.device) * 1e9  # 자기 자신 거리 제거

        min_dist = torch.min(dists)

        if min_dist >= min_distance:
            brightness = torch.ones(1, N, dtype=torch.float32, device=xy_T.device)
            params = torch.cat([xy_T.t(), brightness/3], dim=0)  # [3, N]
            return params.to(device).clone().detach().requires_grad_(True) # Only when gradient connection should be restarted.

    print("Warning: Could not satisfy minimum distance constraint.")
    brightness = torch.ones(1, N, dtype=torch.float32)
    params = torch.cat([xy_T.t(), brightness], dim=0)
    return params.to(device).clone().detach().requires_grad_(True)

def Params_visualization(params_gt, params_est, Num_pixel, bias, beta=None):
    """
    GT와 추정 파라미터를 시각화용으로 정렬하고 매칭합니다.
    위치(x,y)와 밝기(brightness)를 모두 고려한 자동 가중치 기반 매칭.

    Args:
        params_gt: shape (3, N_gt), tf.Tensor or np.ndarray
        params_est: shape (3, N_est), tf.Tensor or np.ndarray
        Num_pixel: 시각화 좌표 보정용 기준
        bias: 시각화 보정값
        beta: 밝기 weight, None이면 자동 설정됨 (위치 weight alpha=1.0 고정)

    Returns:
        VisualCoord_gt: (3, N_total)
        VisualCoord_est: (3, N_total)
        match_pairs: list of (gt_idx, est_idx)
    """

    alpha = 1.0  # 위치 가중치는 기본값 1.0 고정

    # Convert to numpy if tensor
    if torch.is_tensor(params_gt):
        params_gt = params_gt.detach().cpu().numpy()
    if torch.is_tensor(params_est):
        params_est = params_est.detach().cpu().numpy()

    N_gt = params_gt.shape[1]
    N_est = params_est.shape[1]

    gt_coords = params_gt[0:2].T  # shape: (N_gt, 2)
    est_coords = params_est[0:2].T  # shape: (N_est, 2)

    xy_dists = np.linalg.norm(gt_coords[:, None, :] - est_coords[None, :, :], axis=-1)  # (N_gt, N_est)
    mean_xy_dist = np.mean(xy_dists)

    b_diffs = np.abs(params_gt[2, :, None] - params_est[2, None, :])  # (N_gt, N_est)
    mean_b_diff = np.mean(b_diffs)

    if beta is None:
        beta = (mean_xy_dist / mean_b_diff) if mean_b_diff != 0 else 0.0

    # === Cost matrix ===
    cost_matrix = np.zeros((N_gt, N_est))
    for i in range(N_gt):
        for j in range(N_est):
            xy_dist = np.linalg.norm(params_gt[0:2, i] - params_est[0:2, j])
            b_diff = abs(params_gt[2, i] - params_est[2, j])
            cost_matrix[i, j] = alpha * xy_dist + beta/2 * b_diff

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # === Index sets ===
    matched_gt_indices = set(row_ind)
    matched_est_indices = set(col_ind)

    all_gt_indices = set(range(N_gt))
    all_est_indices = set(range(N_est))

    unmatched_gt_indices = list(all_gt_indices - matched_gt_indices)
    unmatched_est_indices = list(all_est_indices - matched_est_indices)

    # === GT Coordinates ===
    y_gt_matched = Num_pixel // 2 + params_gt[0, row_ind] - bias
    x_gt_matched = Num_pixel // 2 + params_gt[1, row_ind] - bias
    b_gt_matched = params_gt[2, row_ind]

    y_gt_unmatched = Num_pixel // 2 + params_gt[0, unmatched_gt_indices] - bias
    x_gt_unmatched = Num_pixel // 2 + params_gt[1, unmatched_gt_indices] - bias
    b_gt_unmatched = params_gt[2, unmatched_gt_indices]

    y_gt_all = np.concatenate([y_gt_matched, y_gt_unmatched])
    x_gt_all = np.concatenate([x_gt_matched, x_gt_unmatched])
    b_gt_all = np.concatenate([b_gt_matched, b_gt_unmatched])

    VisualCoord_gt = np.stack([y_gt_all, x_gt_all, b_gt_all], axis=0)

    # === Estimate Coordinates ===
    y_est_matched = Num_pixel // 2 + params_est[0, col_ind] - bias
    x_est_matched = Num_pixel // 2 + params_est[1, col_ind] - bias
    b_est_matched = params_est[2, col_ind]

    y_est_unmatched = Num_pixel // 2 + params_est[0, unmatched_est_indices] - bias
    x_est_unmatched = Num_pixel // 2 + params_est[1, unmatched_est_indices] - bias
    b_est_unmatched = params_est[2, unmatched_est_indices]

    y_est_all = np.concatenate([y_est_matched, y_est_unmatched])
    x_est_all = np.concatenate([x_est_matched, x_est_unmatched])
    b_est_all = np.concatenate([b_est_matched, b_est_unmatched])

    VisualCoord_est = np.stack([y_est_all, x_est_all, b_est_all], axis=0)

    match_pairs = list(zip(row_ind, col_ind))

    return VisualCoord_gt, VisualCoord_est, match_pairs

def Reorder_estimate(Best_OptParams_guess, match_pairs):
    """
    Best_OptParams_guess: shape (3, N_est)
    match_pairs: list of (gt_idx, est_idx)
    Returns:
        reordered_est: shape (3, N_matched), reordered to match GT order
    """
    if torch.is_tensor(Best_OptParams_guess):
        Best_OptParams_guess = Best_OptParams_guess.detach().cpu()
    sorted_est_params = np.zeros_like(Best_OptParams_guess[:, :len(match_pairs)])

    for gt_idx, est_idx in match_pairs:
        sorted_est_params[:, gt_idx] = Best_OptParams_guess[:, est_idx]

    return sorted_est_params
#%%


