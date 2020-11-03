import numpy as np
import functools
import warnings
import desc
import os
os.environ["JAX_PLATFORM_NAME"] = 'cpu'


class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    TIMER = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


if os.environ.get('DESC_USE_NUMPY'):
    jnp = np
    use_jax = False
    print('DESC version {}, using numpy backend, version={}, dtype={}'.format(desc.__version__,
                                                                              np.__version__, np.linspace(0, 1).dtype))
else:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jax
            import jaxlib
            import jax.numpy as jnp
            from jax.config import config
            config.update("jax_enable_x64", True)
            x = jnp.linspace(0, 5)
            y = jnp.exp(x)
        use_jax = True
        print('DESC version {}, using JAX backend, jax version={}, jaxlib version={}, dtype={}'.format(
            desc.__version__, jax.__version__, jaxlib.__version__, x.dtype))
    except:
        jnp = np
        use_jax = False
        warnings.warn(TextColors.WARNING +
                      'Failed to load JAX' + TextColors.ENDC)
        print('DESC version {}, using numpy backend, version={}, dtype={}'.format(desc.__version__,
            np.__version__, np.linspace(0, 1).dtype))

if use_jax:
    jit = jax.jit
    fori_loop = jax.lax.fori_loop

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.


        Args:
            arr (array-like): Array to populate
            inds (array-like of int): Indices to populate
            vals (array-like): Values to insert

        Returns:
            arr (array-like). Input array with vals inserted at inds.
        """

        return jax.ops.index_update(arr, inds, vals)

    @jit
    def factorial(n):
        """Factorial function for jax backend

        Args:
            n (int,array-like): input values. if n<0, returns 0

        Returns:
            n! (float): factorial of n

        """
        x = jnp.asarray(n+1)
        y = jnp.exp(jax.scipy.special.gammaln(x))
        y = jnp.where(x < 1, 0, y)
        return y

else:
    jit = lambda func, *args, **kwargs: func
    from scipy.special import factorial

    # we divide by zero in a few places but then overwrite with the
    # correct asmptotic values, so lets suppress annoying warnings about that
    np.seterr(divide='ignore', invalid='ignore')

    def put(arr, inds, vals):
        """Functional interface for array "fancy indexing"

        basically a way to do arr[inds] = vals in a way that plays nice with jit/autodiff.


        Args:
            arr (array-like): Array to populate
            inds (array-like of int): Indices to populate
            vals (array-like): Values to insert

        Returns:
            arr (array-like). Input array with vals inserted at inds.
        """

        arr[inds] = vals
        return arr

    def fori_loop(lower, upper, body_fun, init_val):
        """Loop from lower to upper, applying body_fun to init_val

        This version is for the numpy backend, for jax backend see jax.fori_loop
        The semantics of ``fori_loop`` are given by this Python implementation::

            def fori_loop(lower, upper, body_fun, init_val):
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val
        Args:
            lower: an integer representing the loop index lower bound (inclusive)
            upper: an integer representing the loop index upper bound (exclusive)
            body_fun: function of type ``(int, a) -> a``.
            init_val: initial loop carry value of type ``a``.

        Returns:
            Loop value from the final iteration, of type ``a``.
        """
        val = init_val
        for i in np.arange(lower, upper):
            val = body_fun(i, val)
        return val


class Timer():
    """Simple object for organizing timing info

    Create a Timer object, which can then keep track of 
    multiple concurrent performance timers, each associated with
    a given name. 

    Individual timers can be started and stopped with 
    >>>timer.start(name)
    >>>timer.stop(name)

    The elapsed time can be printed with 
    >>>timer.disp(name)

    Raw values of elapsed time (in seconds) can be retrieved 
    with
    >>>timer[name]
    """

    def __init__(self, ns=True):
        import time
        self._times = {}
        self._timers = {}
        self._ns = ns
        if self._ns:
            try:
                self.op = time.perf_counter_ns
            except AttributeError:
                self.op = time.perf_counter
                self._ns = False
                warnings.warn(TextColors.WARNING +
                              'nanosecond timing not available on this system, reverting to microsecond timing' + TextColors.ENDC)
        else:
            self.op = time.perf_counter

    def start(self, name):
        """Starts a timer

        Args:
            name (str): name to associate with timer
        """

        self._timers[name] = [self.op()]

    def stop(self, name):
        """Stops a running timer:

        Args:
            name (str): name of timer to stop

        Raises:
            ValueError: if timer 'name' has not been started
        """

        try:
            self._timers[name].append(self.op())
        except KeyError:
            raise ValueError(
                TextColors.FAIL + "timer '{}' has not been started".format(name) + TextColors.ENDC) from None
        self._times[name] = np.diff(self._timers[name])[0]
        if self._ns:
            self._times[name] = self._times[name]/1e9
        del self._timers[name]

    @staticmethod
    def pretty_print(name, time):
        """Pretty prints time interval

        Does not modify or use any internal timer data,
        this is just a helper for pretty printing arbitrary time data

        Args:
            name (str): text to print before time
            time (float): time (in seconds) to print
        """
        us = time*1e6
        ms = us / 1000
        sec = ms / 1000
        mins = sec / 60
        hrs = mins / 60

        if us < 100:
            out = '{:.3f}'.format(us)[:4] + ' us'
        elif us < 1000:
            out = '{:.3f}'.format(us)[:3] + ' us'
        elif ms < 100:
            out = '{:.3f}'.format(ms)[:4] + ' ms'
        elif ms < 1000:
            out = '{:.3f}'.format(ms)[:3] + ' ms'
        elif sec < 60:
            out = '{:.3f}'.format(sec)[:4] + ' sec'
        elif mins < 60:
            out = '{:.3f}'.format(mins)[:4] + ' min'
        else:
            out = '{:.3f}'.format(hrs)[:4] + ' hrs'

        print(TextColors.TIMER + 'Timer: {} = {}'.format(name, out) + TextColors.ENDC)

    def disp(self, name):
        """Pretty prints elapsed time

        If the timer has been stopped, it reports the time delta between
        start and stop. If it has not been stopped, it reports the current
        elapsed time and keeps the timing running.

        Args:
            name (str): name of the timer to display

        Raises:
            ValueError: if timer 'name' has not been started
        """

        try:     # has the timer been stopped?
            time = self._times[name]
        except KeyError:  # might still be running, let's check
            try:
                start = self._timers[name][0]
                now = self.op()   # don't stop it, just report current elapsed time
                time = float(now-start)/1e9 if self._ns else (now-start)
            except KeyError:
                raise ValueError(
                    TextColors.FAIL + "timer '{}' has not been started".format(name) + TextColors.ENDC) from None

        self.pretty_print(name, time)

    def __getitem__(self, key):
        return self._times[key]

    def __setitem__(self, key, val):
        self._times[key] = val


class _Indexable():
    """Helper object for building indexes for indexed update functions.
    This is a singleton object that overrides the ``__getitem__`` method
    to return the index it is passed.
    >>> opsindex[1:2, 3, None, ..., ::2]
    (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))

    copied from jax.ops.index to work with either backend
    """
    __slots__ = ()

    def __getitem__(self, index):
        return index


"""
Helper object for building indexes for indexed update functions.
This is a singleton object that overrides the ``__getitem__`` method
to return the index it is passed.
>>> opsindex[1:2, 3, None, ..., ::2]
(slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))

copied from jax.ops.index to work with either backend
"""
opsindex = _Indexable()


def conditional_decorator(dec, condition, *args, **kwargs):
    """Apply arbitrary decorator to a function if condition is met

    Args:
        dec (decorator): Decorator to apply
        condition (bool): condition that must be met for decorator to be applied
        args: Arguments to pass to decorator
        kwargs: Keyword arguments to pass to decorator

    Returns:
       cond_dec (decorator): Decorator that acts like ``dec`` if ``condition``, 
       otherwise does nothing.
    """
    @functools.wraps(dec)
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func, *args, **kwargs)
    return decorator


def issorted(x, axis=None, tol=1e-12):
    """Checks if an array is sorted, within a given tolerance

    Checks whether x[i+1] - x[i] > tol

    Args:
        x (array-like): input values
        axis (int): axis along which to check if the array is sorted.
            If None, the flattened array is used.
        tol (float): tolerance for determining order. Array is still considered sorted
            if the difference between adjacent values is greater than -tol

    Returns:
        bool: whether the array is sorted
    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(np.diff(x, axis=axis) >= -tol)


def isalmostequal(x, axis=-1, tol=1e-12):
    """Checks if all values of an array are equal, to within a given tolerance

    Args:
        x (array-like): input values
        axis (int): axis along which to make comparison. If None, the flattened array is used
        tol (float): tolerance for comparison. Array is considered equal if std(x)*len(x)< tol along axis

    Returns:
        bool: whether the array is equal
    """
    if axis is None:
        x = x.flatten()
        axis = 0
    return np.all(x.std(axis=axis)*x.shape[axis] < tol)


def dot(a, b, axis):
    """Batched vector dot product

    Args:
        a (array-like): first array of vectors
        b (array-like): second array of vectors        
        axis (int): axis along which vectors are stored

    Returns:
        y (array-like): y = sum(a*b, axis=axis)
    """
    return jnp.sum(a*b, axis=axis, keepdims=False)


def sign(x):
    """Sign function, but returns 1 for x==0

    Args:
        x (array-like): array of input values

    Returns 
        y (array-like): 1 where x>=0, -1 where x<0
    """
    x = jnp.atleast_1d(x)
    y = jnp.where(x == 0,1,jnp.sign(x))
    return y


def cross(a, b, axis):
    """Batched vector cross product

    Args:
        a (array-like): first array of vectors
        b (array-like): second array of vectors        
        axis (int): axis along which vectors are stored

    Returns:
        y (array-like): y = a x b
    """
    return jnp.cross(a, b, axis=axis)


def rms(x):
    """Compute rms value of an array

    Args:
        x (array-like): input array

    Returns:
        y (float): rms value of x, eg sqrt(sum(x**2))
    """
    return jnp.sqrt(jnp.mean(x**2))


def iotafun(rho, nu, params):
    """Rotational transform

    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile

    Returns:
        iota (array-like): iota profile (or derivative) evaluated at rho
    """
    return jnp.polyval(jnp.polyder(params[::-1], nu), rho)


def presfun(rho, nu, params):
    """Plasma pressure

    Args:
        rho (array-like): coordinates at which to evaluate
        nu (int): order of derivative (for compatibility with scipy spline routines)
        params (array-like): parameters to use for calculating profile

    Returns:
        pres (array-like): pressure profile (or derivative) evaluated at rho
    """
    return jnp.polyval(jnp.polyder(params[::-1], nu), rho)


def get_needed_derivatives(mode, axis=True):
    """Get array of derivatives needed for calculating objective function

    Args:
        mode (str): one of ``None``, ``'force'``, ``'accel'``, ``'qs'``, or ``'all'``
        axis (bool): whether to include terms needed for axis expansion

    Returns:
        derivs (array, shape (N,3)): combinations of derivatives of R,Z needed
            to compute objective function. Each row is one set, columns represent
            the order of derivative for [rho, theta, zeta].
    """
    equil_derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                             [0, 1, 1], [0, 0, 2]])
    axis_derivs = np.array([[2, 1, 0], [1, 2, 0], [1, 1, 1], [2, 2, 0]])
    qs_derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                          [2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0],
                          [0, 1, 1], [0, 0, 2], [3, 0, 0], [2, 1, 0],
                          [2, 0, 1], [1, 2, 0], [1, 1, 1], [1, 0, 2],
                          [0, 3, 0], [0, 2, 1], [0, 1, 2], [0, 0, 3],
                          [2, 2, 0]])
    if mode is None:
        return np.array([[0, 0, 0]])
    elif mode.lower() in ['force', 'accel']:
        if axis:
            return np.vstack([equil_derivs, axis_derivs])
        else:
            return equil_derivs
    elif mode.lower() in ['all', 'qs']:
        return qs_derivs
    else:
        raise NotImplementedError(
            TextColors.FAIL + "derivs must be one of 'force', 'accel', 'all', 'qs'" + TextColors.ENDC)


def unpack_x(x, nRZ):
    """Unpacks the optimization state vector x into cR,cZ,cL components

    Args:
        x (ndarray): vector to unpack
        nRZ (int): number of R,Z coeffs        

    Returns:
        cR (ndarray, shape(N_coeffs,)): spectral coefficients of R
        cZ (ndarray, shape(N_coeffs,)): spectral coefficients of Z
        cL (ndarray, shape(2M+1)*(2N+1)): spectral coefficients of lambda           
    """

    cR = x[:nRZ]
    cZ = x[nRZ:2*nRZ]
    cL = x[2*nRZ:]
    return cR, cZ, cL


class FiniteDifferenceJacobian():
    """Class that wraps a function and computes its jacobian using 2nd order centered finite differences

    Args:
        fun (callable): function to wrap
        rel_step (float): relative step size for finite differences. 
            step_size = rel_step * x0 * max(1,abs(x0))

    Returns:
       jac_fun (callable): object that computes the jacobian of fun.
    """

    def __init__(self, fun, rel_step=np.finfo(np.float64).eps**(1/3), **kwargs):
        self.fun = fun
        self.rel_step = rel_step

    def __call__(self, x0, *args):
        """Evaluate the jacobian of fun at x0.

        Args:
            x0 (array-like): point to evaluate jacobian
            args: additional arguments passed to fun.

        Returns:
            dF/dx (array-like): Jacobian of fun at x0.
        """
        f0 = self.fun(x0, *args)
        m = f0.size
        n = x0.size
        J_transposed = np.empty((n, m))
        idx = np.arange(m).astype(jnp.int64)
        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = self.rel_step * sign_x0 * np.maximum(1.0, jnp.abs(x0))
        h_vecs = np.diag(h)
        for i in range(h.size):
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            f1 = self.fun(x1, *args)
            f2 = self.fun(x2, *args)
            df = f2 - f1
            dfdx = df / dx
            J_transposed = put(J_transposed, i*m+idx, dfdx)
        if m == 1:
            J_transposed = np.ravel(J_transposed)
        return J_transposed.T


class SPSAJacobian():
    """Class for computing jacobian simultaneous perturbation stochastic approximation

    Args:
        fun (callable): function to be differentiated
        rel_step (float): relative step size for finite difference
        N (int): number of samples to take
    """

    def __init__(self, fun, rel_step=1e-6, N=100, **kwargs):

        self.fun = fun
        self.rel_step = rel_step
        self.N = N

    def __call__(self, x0, *args, **kwargs):
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

    Args:
        fun (callable): function to be differentiated
        x0 (array-like): starting point
        f0 (array-like): function evaluated at starting point
        J0 (array-like): estimate of jacobian at starting point
            If not given, the identity matrix is used
        minstep (float): minimum step size for updating the jacobian
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

    def __call__(self, x, *args, **kwargs):
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

    Args:
        fun (callable): function to take jacobian of
        N (int): dimension of fun(x)
        M (int): dimension of x
        block_size (int): size (number of rows) of each block.
            the last block may be smaller depending on N and
        num_blocks (int): number of blocks (only used if block size
            is not given).
        devices (jax.device, list, tuple): list of jax devices to use.
            Blocks will be split evenly across them.
        jit (bool): whether to apply JIT compilation. Generally
            only worth if it if jacobian will be called many times
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

    def __call__(self, x, *args):

        return np.vstack([jac(x, *args) for jac in self.jac_blocks])


@conditional_decorator(functools.partial(jit), use_jax)
def polyder_vec(p, m):
    """Vectorized version of polyder for differentiating multiple polynomials of the same degree

    Args:
        p (ndarray, shape(N,M)): polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
            each column is a power of x
        m (int >=0): order of derivative

    Returns:
        der (ndarray, shape(N,M)): polynomial coefficients for derivative in descending order
    """
    m = jnp.asarray(m, dtype=int)  # order of derivative
    p = jnp.atleast_2d(p)
    l = p.shape[0]               # number of polynomials
    n = p.shape[1] - 1           # order of polynomials

    D = jnp.arange(n, -1, -1)
    D = factorial(D)/factorial(D-m)

    p = jnp.roll(D*p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

    return p


@conditional_decorator(functools.partial(jit), use_jax)
def polyval_vec(p, x):
    """Evaluate a polynomial at specific values, 
    vectorized for evaluating multiple polynomials of the same degree.

    Parameters:
        p (ndarray, shape(N,M)): Array of coefficient for N polynomials of order M. 
            Each row is one polynomial, given in descending powers of x. 
        x (array-like, len(K,)): A number, or 1d array of numbers at
            which to evaluate p. If greater than 1d it is flattened.

    Returns:
        y (ndarray, shape(N,K)): polynomials evaluated at x.
            Each row corresponds to a polynomial, each column to a value of x

    Notes:
        Horner's scheme is used to evaluate the polynomial. Even so,
        for polynomials of high degree the values may be inaccurate due to
        rounding errors. Use carefully.
    """
    p = jnp.atleast_2d(p)
    npoly = p.shape[0]
    order = p.shape[1]
    x = jnp.asarray(x).flatten()
    nx = len(x)
    y = jnp.zeros((npoly, nx))

    def body_fun(k, y):
        return y * x + p[:, k][:, jnp.newaxis]
    y = fori_loop(0, order, body_fun, y)

    return y


if use_jax:
    jacfwd = jax.jacfwd
    jacrev = jax.jacrev
    grad = jax.grad
else:
    jacfwd = FiniteDifferenceJacobian
    jacrev = FiniteDifferenceJacobian
    grad = FiniteDifferenceJacobian
