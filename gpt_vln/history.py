"""Chronological uniform sampling, matching the original GPT6-VLN strategy."""


def uniform_history_indices(total, count=8):
    if total < 0 or count < 0:
        raise ValueError('total and count must be nonnegative')
    if count == 0 or total == 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    return [i * (total - 1) // (count - 1) for i in range(count)]
