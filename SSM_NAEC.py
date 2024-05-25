# Xianrui(Henry) Wang
# 2024-5-9

import numpy as np
from utils import multichannel_stft
from STFTConfig import n_fft, hop
import librosa
import time


def _Construct_P_Order_Signal(ref_T, mic_T, P):
    # calculate the matrix consisting of P order far-end signals
    xp_TP = np.zeros((ref_T.shape[0], P))
    for p_idx in range(P):
        xp_TP[:, p_idx] = np.power(ref_T, 2 * p_idx + 1)
    # Construct the observed signal, Q=P+1
    yxp_QT = np.transpose(np.concatenate((mic_T[:, None], xp_TP), axis=1))
    # transform into STFT domain
    Y_tilde_QFT = multichannel_stft(yxp_QT, n_fft, hop).transpose([2, 0, 1])
    # separate two parts
    # microphone signal
    Y_FT = Y_tilde_QFT[0, :, :]
    X_PFT = Y_tilde_QFT[1:, :, :]
    X_FPT = X_PFT.swapaxes(0, 1)
    return Y_FT, X_FPT


def SSM_NAEC(ref_T, mic_T, p, L, A=0.99997, alpha_e=0.992):
    """
    Signal Channel Nonlinear acoustic echo cancellation based on State-Space-Model(SSM) and Convolutive Transfer Function
    Single channel version of:
    J. Park and J.-H Chang, "State-Space Microphone Array Nonlinear Acoustic Echo Cancellation Using Multiple Near-END
    Speech Covariance," IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    vol. 27, no. 10, pp. 1520-1534, 2019.
    """
    if not ref_T.shape == mic_T.shape:
        raise ValueError("the reference signal should have the same length with microphone signal")
    pass
    Y_FT, X_FPT = _Construct_P_Order_Signal(ref_T, mic_T, p)
    # initialization
    n_freq, n_frame = Y_FT.shape
    sHat_FT = np.zeros_like(Y_FT)
    h_PFL = np.zeros((p, n_freq, L)).astype(Y_FT.dtype)
    Phi_h_PFLL = np.tile(1e-2*np.eye(L), (p, n_freq, 1, 1))
    X_PFT = X_FPT.swapaxes(0, 1)
    Phi_w_PFLL = np.tile(1e-2*np.eye(L), (p, n_freq, 1, 1))
    Phi_e_PF = 1e-2*np.ones((p, n_freq)).astype(Y_FT.dtype)
    X_buff_PFL = np.zeros((p, n_freq, L)).astype(Y_FT.dtype)
    time1 = time.time()
    for n_idx in range(n_frame):
        # for each frame, shift the signal into the buffer
        #print(X_buff_PFL.shape)
        X_buff_PFL[:, :, 1:] = X_buff_PFL[:, :, :-1]
        X_buff_PFL[:, :, 0] = X_PFT[:, :, n_idx]
        Y_F = Y_FT[:, n_idx]
        # prediction
        h_plus_PFL = A * h_PFL
        echo_plus_F = np.sum((X_buff_PFL * h_plus_PFL), axis=(0, 2), keepdims=False)
        Phi_h_plus_PFLL = (A**2.0) * Phi_h_PFLL + Phi_w_PFLL
        # update
        K_PFL = (np.squeeze(Phi_h_plus_PFLL @ np.conj(X_buff_PFL[:, :, :, None])) / (np.squeeze(X_buff_PFL[:, :, None, :]
                            @ Phi_h_plus_PFLL @ np.conj(X_buff_PFL[:, :, :, None])) + Phi_e_PF + 1e-8)[:, :, None])
        h_PFL_new = h_plus_PFL + np.squeeze(K_PFL * (Y_F - echo_plus_F)[None, :, None])
        Phi_w_PFLL = (1/L * np.mean(np.abs(h_PFL_new - h_PFL) ** 2, axis=-1, keepdims=False)[:, :, None, None]
                      * np.tile(np.eye(L), (p, n_freq, 1, 1)))
        h_PFL = h_PFL_new
        Phi_h_PFLL = (np.tile(np.eye(L), (p, n_freq, 1, 1)) - K_PFL[:, :, :, None] @
                      X_buff_PFL[:, :, None, :]) @ Phi_h_plus_PFLL
        # output
        echo_F = np.sum((X_buff_PFL * h_PFL), axis=(0, 2), keepdims=False)
        # update the error covariance matrix
        Phi_e_PF = alpha_e * Phi_e_PF + (1 - alpha_e) * ((Y_F - echo_F) * np.conj(Y_F - echo_F))[None, :]
        sHat_FT[:, n_idx] = Y_F - echo_F
    time2 = time.time()
    sHat_T = librosa.istft(sHat_FT, win_length=n_fft, hop_length=hop)
    return sHat_T, time2-time1


