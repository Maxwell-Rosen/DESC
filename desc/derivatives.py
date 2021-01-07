import numpy as np
from abc import ABC, abstractmethod

from desc.backend import TextColors, use_jax, put

if use_jax:
    import jax


class _Derivative(ABC):
    """_Derivative is an abstract base class for derivative matrix calculations
    """

    @abstractmethod
    def __init__(self, fun: callable, argnum: int = 0, **kwargs) -> None:
        """Initializes a Derivative object

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnums : int, optional
            Specifies which positional argument to differentiate with respect to

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute(self, *args):
        """Computes the derivative matrix

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """
        pass

    @property
    def fun(self) -> callable:
        return self._fun

    @fun.setter
    def fun(self, fun: callable) -> None:
        self._fun = fun

    @property
    def argnum(self) -> int:
        return self._argnum

    @argnum.setter
    def argnum(self, argnum: int) -> None:
        self._argnum = argnum

    def __call__(self, *args):
        return self.compute(*args)


class AutoDiffDerivative(_Derivative):
    """Computes derivatives using automatic differentiation with JAX
    """

    def __init__(self, fun: callable, argnum: int = 0, mode: str = 'fwd') -> None:
        """Initializes an AutoDiffDerivative

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnums : int, optional
            Specifies which positional argument to differentiate with respect to
        mode : str, optional
            Automatic differentiation mode.
            One of 'fwd' (forward mode jacobian), 'rev' (reverse mode jacobian),
            'grad' (gradient of a scalar function), or 'hess' (hessian of a scalar function)
            Default = 'fwd'

        Raises
        ------
        ValueError

        Returns
        -------
        None

        """
        self._fun = fun
        self._argnum = argnum
        self.mode = mode

    def compute(self, *args):
        """Computes the derivative matrix

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """
        return self._compute(*args)

    def _compute_jvp(self, v, *args):
        tangents = list(args)
        tangents[self.argnum] = v
        y = jax.jvp(self._fun, args, tangents)
        return y

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in ['fwd', 'rev', 'grad', 'hess', 'jvp']:
            raise ValueError(TextColors.FAIL +
                             "invalid mode option for automatic differentiation"
                             + TextColors.ENDC)
        self._mode = mode
        if self._mode == 'fwd':
            self._compute = jax.jacfwd(self._fun, self._argnum)
        elif self._mode == 'rev':
            self._compute = jax.jacrev(self._fun, self._argnum)
        elif self._mode == 'grad':
            self._compute = jax.grad(self._fun, self._argnum)
        elif self._mode == 'hess':
            self._compute = jax.hessian(self._fun, self._argnum)
        elif self._mode == 'jvp':
            self._compute = self._compute_jvp


class FiniteDiffDerivative(_Derivative):
    """Computes derivatives using 2nd order centered finite differences
    """

    def __init__(self, fun: callable, argnum: int = 0, mode: str = 'fwd',
                 rel_step: float = 1e-3) -> None:
        """Initializes a FiniteDiffDerivative

        Parameters
        ----------
        fun : callable
            Function to be differentiated.
        argnums : int, optional
            Specifies which positional argument to differentiate with respect to
        rel_step : float, optional
            Relative step size: dx = max(1, abs(x))*rel_step
            Default = 1e-3

        Returns
        -------
        None

        """
        self._fun = fun
        self._argnum = argnum
        self.rel_step = rel_step
        self.mode = mode

    def _compute_hessian(self, *args):
        """Computes the hessian matrix using 2nd order centered finite differences.

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        H : ndarray of float, shape(len(x),len(x))
            d^2f/dx^2, where f is the output of the function fun and x is the input
            argument at position argnum.

        """

        def f(x):
            tempargs = args[0:self._argnum] + \
                (x,) + args[self._argnum+1:]
            return self._fun(*tempargs)

        x = np.atleast_1d(args[self._argnum])
        n = len(x)
        fx = f(x)
        h = np.maximum(1.0, np.abs(x))*self.rel_step
        ee = np.diag(h)
        dtype = fx.dtype
        hess = np.outer(h, h)

        for i in range(n):
            eei = ee[i, :]
            hess[i, i] = (f(x + 2 * eei) - 2 * fx +
                          f(x - 2 * eei)) / (4. * hess[i, i])
            for j in range(i + 1, n):
                eej = ee[j, :]
                hess[i, j] = (f(x + eei + eej)
                              - f(x + eei - eej)
                              - f(x - eei + eej)
                              + f(x - eei - eej)) / (4. * hess[j, i])
                hess[j, i] = hess[i, j]

        return hess

    def _compute_grad_or_jac(self, *args):
        """Computes the gradient or jacobian matrix (ie, first derivative)

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """
        def f(x):
            tempargs = args[0:self._argnum] + (x,) + args[self._argnum+1:]
            return self._fun(*tempargs)

        x0 = np.atleast_1d(args[self._argnum])
        f0 = f(x0)
        m = f0.size
        n = x0.size
        J = np.zeros((m, n))
        h = np.maximum(1.0, np.abs(x0))*self.rel_step
        h_vecs = np.diag(np.atleast_1d(h))
        for i in range(n):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = f(x1)
            f2 = f(x2)
            df = f2 - f1
            dfdx = df / dx
            J = put(J.T, i, dfdx.flatten()).T
        if m == 1:
            J = np.ravel(J)
        return J

    def _compute_jvp(self, v, *args):

        normv = np.linalg.norm(v)
        vh = v/normv
        x = args[self.argnum]

        def f(x):
            tempargs = args[0:self._argnum] + (x,) + args[self._argnum+1:]
            return self._fun(*tempargs)

        h = self.rel_step
        df = (f(x + h*vh) - f(x - h*vh))/(2*h)
        return df * normv

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in ['fwd', 'rev', 'grad', 'hess', 'jvp']:
            raise ValueError(TextColors.FAIL +
                             "invalid mode option for finite difference differentiation"
                             + TextColors.ENDC)
        self._mode = mode
        if self._mode == 'fwd':
            self._compute = self._compute_grad_or_jac
        elif self._mode == 'rev':
            self._compute = self._compute_grad_or_jac
        elif self._mode == 'grad':
            self._compute = self._compute_grad_or_jac
        elif self._mode == 'hess':
            self._compute = self._compute_hessian
        elif self._mode == 'jvp':
            self._compute = self._compute_jvp

    def compute(self, *args):
        """Computes the derivative matrix

        Parameters
        ----------
        *args : list
            Arguments of the objective function where the derivative is to be
            evaluated at.

        Returns
        -------
        J : ndarray of float, shape(len(f),len(x))
            df/dx, where f is the output of the function fun and x is the input
            argument at position argnum.

        """
        return self._compute(*args)


if use_jax:
    Derivative = AutoDiffDerivative
else:
    Derivative = FiniteDiffDerivative


# these classes currently do not meet the Derivative API -----------------------


class SPSAJacobian():
    """Class for computing jacobian simultaneous perturbation stochastic approximation
    Parameters
    ----------
    fun : callable
        function to be differentiated
    rel_step : float
        relative step size for finite difference (Default value = 1e-6)
    N : int
        number of samples to take (Default value = 100)
    Returns
    -------
    jac_fun : callable
        object that computes the jacobian of fun.
    """

    def __init__(self, fun, rel_step=1e-6, N=100, **kwargs):

        self.fun = fun
        self.rel_step = rel_step
        self.N = N

    def compute(self, x0, *args, **kwargs):
        """Update and get the jacobian"""

        f0 = self.fun(x0, *args)
        m = f0.size
        n = x0.size

        J = np.zeros((m, n))
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = self.rel_step * sign_x0 * np.maximum(1.0, np.abs(x0))

        for i in range(self.N):
            dx = (np.random.binomial(1, .5, x0.shape)*2-1)*h
            x1 = x0 + dx
            x2 = x0 - dx
            dx = (x1 - x2).flatten()[np.newaxis]
            f1 = np.atleast_1d(self.fun(x1, *args))
            f2 = np.atleast_1d(self.fun(x2, *args))
            df = (f1-f2).flatten()[:, np.newaxis]
            dfdx = df/dx
            J += dfdx
        return J/self.N


class BroydenJacobian():
    """Class for computing jacobian using rank 1 updates
    Parameters
    ----------
    fun : callable
        function to be differentiated
    x0 : array-like
        starting point
    f0 : array-like
        function evaluated at starting point
    J0 : array-like
        estimate of jacobian at starting point
        If not given, the identity matrix is used
    minstep : float
        minimum step size for updating the jacobian (Default value = 1e-12)
    Returns
    -------
    jac_fun : callable
        object that computes the jacobian of fun.
    """

    def __init__(self, fun, x0, f0, J0=None, minstep=1e-12, **kwargs):

        self.fun = fun
        self.x0 = x0
        self.f0 = f0
        self.shape = (f0.size, x0.size)
        self.J = J0 if J0 is not None else np.eye(*self.shape)
        self.minstep = minstep
        self.x1 = self.x0
        self.f1 = self.f0

    def compute(self, x, *args, **kwargs):
        """Update and get the jacobian"""

        self.x0 = self.x1
        self.f0 = self.f1
        self.x1 = x
        dx = self.x1-self.x0
        step = np.linalg.norm(dx)
        if step < self.minstep:
            return self.J
        else:
            self.f1 = self.fun(x, *args)
            df = self.f1 - self.f0
            update = (df - self.J.dot(dx))/step**2
            update = update[:, np.newaxis]*dx[np.newaxis, :]
            self.J += update
            return self.J


class BlockJacobian():
    """Computes a jacobian matrix in smaller blocks.
    Takes a large jacobian and splits it into smaller blocks
    (row-wise) for easier computation, possibly allowing each
    block to be computed independently on different devices in
    parallel. Also helps to reduce memory load, allowing
    computation of larger jacobians on limited memory GPUs
    Parameters
    ----------
    fun : callable
        function to take jacobian of
    N : int
        dimension of fun(x)
    M : int
        dimension of x
    block_size : int
        size (number of rows) of each block.
        the last block may be smaller depending on N and block_size
    num_blocks : int
        number of blocks (only used if block size
        is not given).
    devices : list or tuple of jax.device
        list of jax devices to use.
        Blocks will be split evenly across them.
    usejit : bool
        whether to apply JIT compilation. Generally
        only worth it if jacobian will be called many times
    Returns
    -------
    jac_fun : callable
        object that computes the jacobian of fun.
    """

    def __init__(self, fun, N, M, block_size=None, num_blocks=None,
                 devices=None, usejit=False):

        self.fun = fun
        self.N = N
        self.M = M

        # could probably add some fancier logic here to look at M as well when deciding how
        # to split blocks? Though we can't really split the jacobian columnwise without a lot
        # of surgery on the objective function
        if block_size is not None and num_blocks is not None:
            raise ValueError(TextColors.FAIL +
                             "can specify either block_size or num_blocks, not both" + TextColors.ENDC)
        elif block_size is None and num_blocks is None:
            self.block_size = N
            self.num_blocks = 1
        elif block_size is not None:
            self.block_size = block_size
            self.num_blocks = np.ceil(N/block_size).astype(int)
        else:
            self.num_blocks = num_blocks
            self.block_size = np.ceil(N/num_blocks).astype(int)

        if type(devices) in [list, tuple]:
            self.devices = devices
        else:
            self.devices = [devices]

        self.usejit = usejit
        self.f_blocks = []
        self.jac_blocks = []

        for i in range(self.num_blocks):
            # need the i=i in the lambda signature, otherwise i is scoped to
            # the loop and get overwritten, making each function compute the same subset
            self.f_blocks.append(
                lambda x, *args, i=i: self.fun(x, *args)[i*self.block_size:(i+1)*self.block_size])
            # need to use jacrev here to actually get memory savings
            # (plus, these blocks should be wide and short)
            if self.usejit:
                self.jac_blocks.append(
                    jit(jacrev(self.f_blocks[i]), device=self.devices[i % len(self.devices)]))
            else:
                self.jac_blocks.append(jacrev(self.f_blocks[i]))

    def compute(self, x, *args):

        return np.vstack([jac(x, *args) for jac in self.jac_blocks])
