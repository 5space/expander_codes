import numpy as np

def noisy_channel(X: np.ndarray, bit_error: float) -> np.ndarray:
    # Generate random noise (weighted by bit_error)
    noise = np.random.rand(*X.shape) < bit_error
    # Flip bits where noise is True
    Y = np.bitwise_xor(X, noise.astype(int))
    return Y

def noisy_channel_fixed_errors(X: np.ndarray, bit_fraction: float) -> np.ndarray:
    # Generate random indices for errors
    error_indices = np.random.choice(X.size, int(bit_fraction * X.size), replace=False)
    Y = X.copy()
    Y[error_indices] = (Y[error_indices] + 1) % 2
    return Y

def run_code(encoder: callable, channel: callable, decoder: callable, bit_error: float, m: int, runs=1) -> None:

    total = 0
    for i in range(runs):
        plaintext = np.random.randint(0, 2, size=(m, 1))
        ciphertext = encoder(plaintext)
        decoded = decoder(
            channel(ciphertext, bit_error)
        )

        if np.all(ciphertext == decoded):
            total += 1
        print(i, end="\r")
    
    return total/runs