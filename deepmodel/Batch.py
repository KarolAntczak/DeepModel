import numpy as np


def get_next_batch(train_dataset, step, batch_size):
    offset = 0 if batch_size == 1 else (step * batch_size) % (len(train_dataset) - batch_size)
    return train_dataset[offset:offset + batch_size]


def get_next_batch_noised(train_dataset, step, batch_size, noise_ratio=0.5):
    batch = get_next_batch(train_dataset, step, batch_size)
    noise = np.random.choice([0, 1], size=batch.shape, p=[1.0-noise_ratio, noise_ratio])
    return batch * noise
