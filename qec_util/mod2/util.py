import numpy as np


def gauss_elimination_rows(a: np.ndarray, skip_last_column: bool = True) -> np.ndarray:
    """
    Performs Gauss elimination to the given GF2 matrix by adding and permutting rows.
    It does not add or permute columns. The structure of the reduced matrix is:

    100**0****            100**0***0
    010**0****            010**0***0
    001**0****     or     001**0***0
    000001****            000001***0
    000000000*            0000000001
    000000000*            0000000000

    depending on ``skip_last_column``.

    Parameters
    ----------
    a
        Binary matrix to be brought to the described form. Its shape must be ``(N, M)``,
        thus ``a`` can be a square or non-square matrix.
    skip_last_column
        If ``True``, does not process the last column of the matrix ``a``.
        This flag is useful for solving a system of linear equations with ``[a|b]``.

    Returns
    -------
    a
        Reduced matrix using Gauss elimination by rows. If parameter ``a`` is a
        ``galois.Array``, the returned ``a`` is also a ``galois.Array``.
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"'a' must be a numpy array, but {type(a)} was given.")
    if len(a.shape) != 2:
        raise TypeError(f"'a' must be a matrix, but a.shape={a.shape} was given.")

    n, m = a.shape
    pivot_row = 0
    for col in range(m - int(bool(skip_last_column))):
        pivot_found = False
        for row in range(pivot_row, n):
            if a[row, col]:
                pivot_found = True
                if row != pivot_row:
                    a[[pivot_row, row]] = a[[row, pivot_row]]
                break

        if not pivot_found:
            # already in the correct form
            continue

        # eliminate entries except pivot
        for row in range(n):
            if a[row, col] and (row != pivot_row):
                a[row] ^= a[pivot_row]

        pivot_row += 1
        if pivot_row == n:
            break

    return a


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns a solution for ``a @ x = b`` using operations in GF2.

    Parameters
    ----------
    a
        Binary matrix of shape ``(N, M)``, thus can be square or non-square.
    b
        Binary vector of shape ``(N,)``.

    Returns
    -------
    x
        A solution for ``a @ x = b``. It has shape ``(M,)``. If ``a`` and/or
        ``b`` are ``galois.Array``, then ``x`` is also a ``galois.Array``.

    Raises
    ------
    ValueError
        If the system does not have a solution.

    Notes
    -----
    This function requires ``galois``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    if not isinstance(a, np.ndarray):
        raise TypeError(f"'a' must be a numpy array, but {type(a)} was given.")
    if not isinstance(b, np.ndarray):
        raise TypeError(f"'b' must be a numpy array, but {type(b)} was given.")
    if len(a.shape) != 2:
        raise TypeError(f"'a' must be a matrix, but a.shape={a.shape} was given.")
    if len(b.shape) != 1:
        raise TypeError(f"'b' must be a vector, but b.shape={b.shape} was given.")
    if a.shape[0] != b.shape[0]:
        raise TypeError("'a' and 'b' must have the same number of rows.")

    import galois

    a_aug = galois.GF2(np.concatenate([a, b.reshape(-1, 1)], axis=1))
    a_red = gauss_elimination_rows(a_aug, skip_last_column=True)

    # Identify pivots and check for inconsistency
    n, m = a.shape
    pivot_columns = []
    for col in range(m):
        pivot = np.zeros(n, dtype=int)
        pivot[len(pivot_columns)] = 1
        if (a_red[:, col] == pivot).all():
            pivot_columns.append(col)
    # number of pivot rows = number of pivot columns
    if a_red[len(pivot_columns) :, -1].any():
        raise ValueError("The given linear system does not have a solution.")

    x = galois.GF2(np.zeros(m, dtype=int))
    x[pivot_columns] = a_red[: len(pivot_columns), -1]

    if not any([isinstance(a, galois.Array), isinstance(b, galois.Array)]):
        x = np.array(x)

    return x


def decompose_into_basis(vector: np.ndarray, basis: np.ndarray):
    """
    Decomposes the given vector in terms of the specified basis vectors, so that
    ``basis @ decomposition = vector``.

    Parameters
    ----------
    vector
        Vector to decompose.
    basis
        Matrix with columns as basis vectors.

    Returns
    -------
    The decomposition of the vector in terms of the basis vectors.

    Raises
    ------
    ValueError
        If the vector cannot be expressed in terms of the basis vectors.

    Notes
    -----
    This function requires ``galois``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    return solve_linear_system(a=basis, b=vector)
