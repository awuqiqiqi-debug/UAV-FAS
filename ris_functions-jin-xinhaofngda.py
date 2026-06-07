import numpy as np
from scipy.special import erfinv
from scipy.linalg import eigh

def db2pow(db):
    """Convert dB to power"""
    return 10 ** (db / 10)

def qfunc(x):
    """Q-function: 0.5 * erfc(x/sqrt(2))"""
    return 0.5 * (1 - erfinv(x / np.sqrt(2)))

def qfuncinv(y):
    """Inverse Q-function"""
    return np.sqrt(2) * erfinv(1 - 2 * y)

def trace(A):
    """Matrix trace"""
    return np.trace(A)

def pow_abs(x, p):
    """Power of absolute value"""
    return np.power(np.abs(x), p)

def CalcR(t, W1, W2, U, a0, afa, Pp, Gs, G1, G2, H1, H2, Q1, Q2, F, M, fs, delta, sigma):
    """Calculate achievable rate"""
    gamma_s = np.real(Pp * trace(Gs.conj().T @ U @ Gs)) / np.real(sigma * trace(F.conj().T @ U @ F) + sigma)
    Pd = qfunc((delta - sigma * (M + gamma_s)) / (sigma * np.sqrt(2 * gamma_s + M) / np.sqrt(t * 1e-3 * fs)))
    a1 = afa * (1 - Pd)
    
    r011 = np.real(trace(G1.conj().T @ U @ G1 @ W1)) / np.real(trace(G1.conj().T @ U @ G1 @ W2) + sigma * trace(H1 @ U) + sigma)
    r111 = np.real(trace(G1.conj().T @ U @ G1 @ W1)) / np.real(trace(G1.conj().T @ U @ G1 @ W2) + Pp * trace(Q1 @ U) + sigma * trace(H1 @ U) + sigma)
    r022 = np.real(trace(G2.conj().T @ U @ G2 @ W2)) / np.real(sigma * trace(H2 @ U) + sigma)
    r122 = np.real(trace(G2.conj().T @ U @ G2 @ W2)) / np.real(Pp * trace(Q2 @ U) + sigma * trace(H2 @ U) + sigma)
    
    SumRate = a0 * (np.log2(1 + r011) + np.log2(1 + r022)) + a1 * (np.log2(1 + r111) + np.log2(1 + r122))
    return SumRate

def ULA(theta, M):
    """Uniform Linear Array response"""
    m = np.arange(M)
    ula_result = np.exp(1j * np.pi * m * np.sin(theta))
    return ula_result

def UPA(phi, theta, Ny, Nz):
    """Uniform Planar Array response"""
    m = np.arange(Ny)
    a_ay = np.exp(1j * np.pi * m * np.sin(phi) * np.sin(theta))
    n = np.arange(Nz)
    a_az = np.exp(1j * np.pi * n * np.cos(theta))
    upa_result = np.kron(a_ay, a_az)
    return upa_result

def initial():
    """Generate angles for channel modeling"""
    ST_loc = np.array([0, 0, 5])
    PT_loc = np.array([0, 50, 5])
    IRS_loc = np.array([0, 30, 2])
    U1 = np.array([5, 38, 0])
    U2 = np.array([5, 30, 0])
    U3 = np.array([5, 50, 0])
    User_Num = 3
    
    SI = np.linalg.norm(IRS_loc - ST_loc)
    PI = np.linalg.norm(IRS_loc - PT_loc)
    IU1 = np.linalg.norm(IRS_loc - U1)
    IU2 = np.linalg.norm(IRS_loc - U2)
    IU3 = np.linalg.norm(IRS_loc - U3)
    d_IU = np.array([[IU1], [IU2], [IU3]])
    
    SId = np.linalg.norm(ST_loc[:2] - IRS_loc[:2])
    SIz = abs(ST_loc[2] - IRS_loc[2])
    SIy = abs(IRS_loc[1] - ST_loc[1])
    SIx = abs(ST_loc[0] - IRS_loc[0])
    
    PId = np.linalg.norm(PT_loc[:2] - IRS_loc[:2])
    PIz = abs(PT_loc[2] - IRS_loc[2])
    PIy = abs(IRS_loc[1] - PT_loc[1])
    PIx = abs(PT_loc[0] - IRS_loc[0])
    
    IU1d = np.linalg.norm(IRS_loc[:2] - U1[:2])
    IU1z = abs(IRS_loc[2] - U1[2])
    IU1y = abs(IRS_loc[1] - U1[1])
    IU1x = abs(IRS_loc[0] - U1[0])
    
    IU2d = np.linalg.norm(IRS_loc[:2] - U2[:2])
    IU2z = abs(IRS_loc[2] - U2[2])
    IU2y = abs(IRS_loc[1] - U2[1])
    IU2x = abs(IRS_loc[0] - U2[0])
    
    IU3d = np.linalg.norm(IRS_loc[:2] - U3[:2])
    IU3z = abs(IRS_loc[2] - U3[2])
    IU3y = abs(IRS_loc[1] - U3[1])
    IU3x = abs(IRS_loc[0] - U3[0])
    
    theta_STI = np.arctan(SId / SIz)
    theta_SIRS = np.arctan(SIz / SId)
    phi_SIRS = np.arctan(SIx / SIy)
    
    theta_IST = np.arctan(SIz / SId)
    theta_IRSS = np.arctan(SId / SIz)
    phi_IRSS = np.arctan(SIy / SIx)
    
    theta_PT = np.arctan(PId / PIz)
    theta_PIRS = np.arctan(PIz / PId)
    phi_PIRS = np.arctan(PIx / PIy)
    
    theta_SU1 = np.arctan(IU1d / IU1z)
    theta_SU2 = np.arctan(IU2d / IU2z)
    theta_SU3 = np.arctan(IU3d / IU3z)
    phi_SU1 = np.arctan(IU1y / IU1x)
    phi_SU2 = np.arctan(IU2y / IU2x)
    phi_SU3 = np.arctan(IU3y / IU3x)
    
    theta_SU = np.array([[theta_SU1], [theta_SU2], [theta_SU3]])
    phi_SU = np.array([[phi_SU1], [phi_SU2], [phi_SU3]])
    
    return {
        'theta_SU': theta_SU,
        'phi_SU': phi_SU,
        'theta_SIRS': theta_SIRS,
        'phi_SIRS': phi_SIRS,
        'theta_STI': theta_STI,
        'theta_IRSS': theta_IRSS,
        'phi_IRSS': phi_IRSS,
        'theta_IST': theta_IST,
        'theta_PIRS': theta_PIRS,
        'phi_PIRS': phi_PIRS,
        'theta_PT': theta_PT,
        'User_Num': User_Num,
        'SI': SI,
        'PI': PI,
        'd_IU': d_IU
    }

def gen_Channel(params):
    """Generate channel matrices"""
    PL0 = db2pow(42)
    d0 = 1
    M = 4
    Ny = 4
    Nz = 4
    N = Ny * Nz
    IRS_Num = N
    Channel_num = 30
    
    alpha_SI = 3.5
    alpha_PI = 3.5
    alpha_IU = 3.5
    
    beta_SI = db2pow(0)
    beta_PI = db2pow(0)
    beta_IU = db2pow(0)
    
    theta_SU = params['theta_SU']
    phi_SU = params['phi_SU']
    theta_SIRS = params['theta_SIRS']
    phi_SIRS = params['phi_SIRS']
    theta_STI = params['theta_STI']
    theta_IRSS = params['theta_IRSS']
    phi_IRSS = params['phi_IRSS']
    theta_IST = params['theta_IST']
    theta_PIRS = params['theta_PIRS']
    phi_PIRS = params['phi_PIRS']
    theta_PT = params['theta_PT']
    User_Num = params['User_Num']
    SI = params['SI']
    PI = params['PI']
    d_IU = params['d_IU']
    
    def PL(d, alpha):
        return PL0 * (d / d0) ** (-alpha)
    
    G = np.zeros((N, M, Channel_num), dtype=complex)
    hl = np.zeros((N, User_Num, Channel_num), dtype=complex)
    f = np.zeros((N, Channel_num), dtype=complex)
    F = np.zeros((N, M, Channel_num), dtype=complex)
    G1 = np.zeros((N, M, Channel_num), dtype=complex)
    G2 = np.zeros((N, M, Channel_num), dtype=complex)
    G3 = np.zeros((N, M, Channel_num), dtype=complex)
    Gs = np.zeros((N, M, Channel_num), dtype=complex)
    H1 = np.zeros((N, N, Channel_num), dtype=complex)
    H2 = np.zeros((N, N, Channel_num), dtype=complex)
    H3 = np.zeros((N, N, Channel_num), dtype=complex)
    Hs = np.zeros((N, N, Channel_num), dtype=complex)
    Q1 = np.zeros((N, N, Channel_num), dtype=complex)
    Q2 = np.zeros((N, N, Channel_num), dtype=complex)
    
    for channel in range(Channel_num):
        G[:, :, channel] = np.sqrt(PL(SI, alpha_SI)) * (
            np.sqrt(beta_SI / (1 + beta_SI)) * (UPA(phi_SIRS, theta_SIRS, Ny, Nz).reshape(-1, 1) @ ULA(theta_STI, M).reshape(1, -1)) +
            np.sqrt(1 / (1 + beta_SI)) * (np.sqrt(1/2) * np.random.randn(N, M) + 1j * np.sqrt(1/2) * np.random.randn(N, M))
        )
        
        for i in range(User_Num):
            hl[:, i, channel] = np.sqrt(PL(d_IU[i][0], alpha_IU)) * (
                np.sqrt(beta_IU / (1 + beta_IU)) * UPA(phi_SU[i][0], theta_SU[i][0], Ny, Nz) +
                np.sqrt(1 / (1 + beta_IU)) * (np.sqrt(1/2) * np.random.randn(N) + 1j * np.sqrt(1/2) * np.random.randn(N))
            )
        
        f[:, channel] = np.sqrt(PL(PI, alpha_PI)) * (
            np.sqrt(beta_PI / (1 + beta_PI)) * (UPA(phi_PIRS, theta_PIRS, Ny, Nz) * ULA(theta_PT, 1)) +
            np.sqrt(1 / (1 + beta_PI)) * (np.sqrt(1/2) * np.random.randn(N) + 1j * np.sqrt(1/2) * np.random.randn(N))
        )
        
        F[:, :, channel] = np.sqrt(PL(SI, alpha_SI)) * (
            np.sqrt(beta_SI / (1 + beta_SI)) * (UPA(phi_IRSS, theta_IRSS, Ny, Nz).reshape(-1, 1) @ ULA(theta_IST, M).reshape(1, -1)) +
            np.sqrt(1 / (1 + beta_SI)) * (np.sqrt(1/2) * np.random.randn(N, M) + 1j * np.sqrt(1/2) * np.random.randn(N, M))
        )
        
        G1[:, :, channel] = np.diag(hl[:, 0, channel].conj()) @ G[:, :, channel]
        G2[:, :, channel] = np.diag(hl[:, 1, channel].conj()) @ G[:, :, channel]
        G3[:, :, channel] = np.diag(hl[:, 2, channel].conj()) @ G[:, :, channel]
        Gs[:, :, channel] = np.diag(f[:, channel]) @ F[:, :, channel]
        H1[:, :, channel] = hl[:, 0, channel].reshape(-1, 1) @ hl[:, 0, channel].conj().reshape(1, -1)
        H2[:, :, channel] = hl[:, 1, channel].reshape(-1, 1) @ hl[:, 1, channel].conj().reshape(1, -1)
        H3[:, :, channel] = hl[:, 2, channel].reshape(-1, 1) @ hl[:, 2, channel].conj().reshape(1, -1)
        Hs[:, :, channel] = f[:, channel].reshape(-1, 1) @ f[:, channel].conj().reshape(1, -1)
        
        diag_hl1 = np.diag(hl[:, 0, channel])
        diag_hl2 = np.diag(hl[:, 1, channel])
        Q1[:, :, channel] = diag_hl1 @ f[:, channel].reshape(-1, 1) @ (diag_hl1 @ f[:, channel].reshape(-1, 1)).conj().T
        Q2[:, :, channel] = diag_hl2 @ f[:, channel].reshape(-1, 1) @ (diag_hl2 @ f[:, channel].reshape(-1, 1)).conj().T
    
    w1 = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
    w2 = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
    W1 = w1 @ w1.conj().T
    W2 = w2 @ w2.conj().T
    
    q = np.exp(1j * 2 * np.pi * np.random.rand(IRS_Num, 1))
    beta = np.ones(IRS_Num)
    for i in range(IRS_Num):
        beta[i] = np.sqrt(1 + 10 * np.random.rand())
    
    return {
        'M': M,
        'N': N,
        'hl': hl,
        'G': G,
        'F': F,
        'f': f,
        'G1': G1,
        'G2': G2,
        'G3': G3,
        'H1': H1,
        'H2': H2,
        'H3': H3,
        'Gs': Gs,
        'Hs': Hs,
        'Q1': Q1,
        'Q2': Q2,
        'W1': W1,
        'W2': W2,
        'w1': w1,
        'w2': w2,
        'q': q,
        'beta': beta,
        'Channel_num': Channel_num
    }