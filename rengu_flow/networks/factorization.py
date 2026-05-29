"""Integer factorization for LoKr (Kronecker) decomposition. No Comfy dependency."""


def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Decompose dimension into two factors (m, n) with n >= m.

    Same semantics as Comfy weight_adapter.base.factorization.
    Used for LoKr weight decomposition.

    Examples (factor -> dimension -> result):
        -1: 128 -> (8, 16), 512 -> (16, 32)
        2:  128 -> (2, 64)
        4:  128 -> (4, 32)
    """
    if factor > 0 and (dimension % factor) == 0 and dimension >= factor**2:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n
