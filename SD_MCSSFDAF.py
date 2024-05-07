import numpy as np
import scipy as sp
from utils import *


def SD_MCSSFDAF(ref_T, mic_T, N=5, M=256, R=64, em_num=1, A=0.9997):
    if not ref_T.shape == mic_T.shape:
        raise ValueError("the reference signal should have the same length with microphone signal")
    # determine the number of blocks
    nB = (len(ref_T)-M) // R + 1
    print(nB)
    endLen = M + (nB-1) * R
    # the length of impulse response
    L = M - R
    # DFT matrix to transform signal into frequency domain, from eq. 6
    F_MM = sp.linalg.dft(M)
    F_MM = np.asarray(F_MM)
    # used for inverse DFT
    F_inv_MM = np.linalg.inv(F_MM)
    # eq. 15
    Q_MR = np.concatenate((np.eye(R), np.zeros((R, L))), axis=1).T
    G_MM = F_MM @ Q_MR @ Q_MR.T @ F_inv_MM

    # the initial period before AEC
    sHat_all = []
    sHat_all = np.concatenate((mic_T[:L], sHat_all))

    def _Recover_Signal(E_M=None):
        # the zero-padded(L zeros padded)
        s_M = F_inv_MM @ E_M
        # the near-end signal without zeros
        s_R = s_M[:R]
        return s_R.real

    underline_W_NM = np.random.normal(size=(N, M))+1j*np.random.normal(size=(N, M))
    X_nonlinear_NM = np.zeros((N, M), dtype=np.complex_)
    underline_C_MNm = np.zeros((M, N*M), dtype=np.complex_)
    underline_P_M = 1e-4*np.random.normal(size=M) + 1j*1e-4*np.random.normal(size=M)
    underline_P_NNM = np.tile(underline_P_M[None, None, :], [N, N, 1])
    Phi_s_M = 1e-4*np.random.normal(size=M) + 1j*1e-4*np.random.normal(size=M)
    Phi_delta_M = 1e-4*np.random.normal(size=M) + 1j*1e-4*np.random.normal(size=M)
    Phi_delta_NM = np.tile(Phi_delta_M[None, :], [N, 1])
    for tau in range(nB):
        if tau % 100 == 0:
            print("dealing with %d iterations" % tau)
        # the block of reference signal in the tau block, eq. (2)
        x_M = ref_T[tau*R: tau*R+M]
        # the block of microphone signal in the tau block, eq. (9)
        y_R = mic_T[(tau-1)*R+M: tau*R+M]
        # transform the microphone signal into frequency domain, eq. (12)
        Y_M = F_MM @ Q_MR @ y_R
        underline_C_NMM = np.zeros((N, M, M), dtype=np.complex_)
        for i in range(N):
            # odd power series expansion, eq. (61)
            x_nonlinear_M = np.power(x_M, 2*i + 1)
            # transform the nonlinear expression into frequency domain, eq. (7)
            #print(x_nonlinear_M[:, None].shape)
            X_nonlinear_NM[i, :] = F_MM @ x_nonlinear_M
            # eq. (18), (20a)
            underline_C_NMM[i, :, :] = G_MM * x_nonlinear_M[None, :]
            underline_C_MNm[:, i*M:(i+1)*M] = G_MM * x_nonlinear_M[None, :]
        for em_idx in range(em_num):
            # predictions
            # eq. 35(a)
            underline_W_plus_NM = A * underline_W_NM
            # eq. 35(b), note only i=j, Phi_tau_ij ~= 0
            underline_P_plus_NNM = A**2.0 * underline_P_NNM
            for i in range(N):
                underline_P_plus_NNM[i, i, :] += Phi_delta_NM[i, :]
            # eq. 35(c)
            # note the index has been reordered
            # print(underline_P_plus_NNM.transpose[1, 0, 2].shape)
            XP_plus_ji_NM = np.sum(X_nonlinear_NM[None, :, :] * underline_P_plus_NNM.transpose([1, 0, 2]), axis=1)
            D_M = np.sum(XP_plus_ji_NM * XP_plus_ji_NM, axis=0)
            """
            D_M = np.zeros(M, dtype=np.complex_)
            for i in range(N):
                for j in range(N):
                    D_M += R/M * X_nonlinear_NM[i] * underline_P_plus_NNM[i, j] * X_nonlinear_NM[j]
            """
            D_M += Phi_s_M
            #D_M[D_M < 1e-4] = 1e-4
            # eq. 35(d)
            K_aux_NM = np.sum(underline_P_NNM[:, :, :] * np.conj(X_nonlinear_NM[None, :, :]), axis=1)
            K_NM = K_aux_NM / D_M[None, :]
            # eq. 35(e)
            E_plus_M = Y_M - np.squeeze(np.sum(underline_C_NMM[:, :, :] @ underline_W_NM[:, :, None], axis=0))
            # eq. 35(f)
            underline_W_NM = underline_W_plus_NM + K_NM * E_plus_M[None, :]
            # eq. 35(g)
            underline_P_NNM = underline_P_plus_NNM - (K_NM[None, :, :] * XP_plus_ji_NM[:, None, :]).swapaxes(0, 1)
            ### M-steps, update the covariance matrix
            """
            S. Malik, G. Enzner, Online Maximum-likelihood learning of time-varying dynamical models in 
            block-frequency-domain, in Proc IEEE ICASSP, 2010.
            """
            underline_P_NmNm = np.zeros((N*M, N*M), dtype=np.complex_)
            for i in range(N):
                for j in range(N):
                    underline_P_NmNm[i*M:(i+1)*M, j*M:(j+1)*M] = np.diag(underline_P_NNM[i, j, :])
            Phi_s_M = E_plus_M * np.conj(E_plus_M) + np.diag(underline_C_MNm @ underline_P_NmNm @ np.conj(underline_C_MNm.T))
            for i in range(N):
                Phi_delta_NM[i] = (1-A**2) * (np.diag(underline_W_NM[i, :, None] @ np.conj(underline_W_NM[None, i]))
                                                    + underline_P_NNM[i, i])
            # Phi_delta_Nm = (1-A**2) * np.diag(underline_W_NM @ np.conj(underline_W_NM.T) + underline_P_NmNm)
            # Phi_delta_NM = np.reshape(Phi_delta_Nm, [N, M])
        E_M = Y_M - np.squeeze(np.sum(underline_C_NMM[:, :, :] @ underline_W_NM[:, :, None], axis=0))
        sHat = _Recover_Signal(E_M)
        sHat_all = np.concatenate((sHat_all, sHat))
    sHat_all = np.concatenate((sHat_all, mic_T[endLen:]))
    return sHat_all




if __name__ == "__main__":
    far_end = np.random.normal(size=16000)
    mic = np.random.normal(size=16000)
    sHat = SD_MCSSFDAF(far_end, mic, A=0.99997)
    print(sHat.shape)
    sf.write("test.wav", sHat, samplerate=16000)


