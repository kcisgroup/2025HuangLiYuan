
import math
import numpy as np
import scipy.special as sp
import itertools as it


# ***************************************************************************************

def num_to_mant_exp( num ):
    """
    This function returns the (base 10) exponent and mantissa of a number.

    :param num: input number.
    :type num: :class:`int` or :class:`float`
    :return: tuple (mantissa, exponent) of :class:`int` containing the mantissa and the exponent of the input number.
    :rtype: tuple

    """
    try:
        exponent = math.floor(math.log10(abs(num)))
    except ValueError:  # Case of log10(0)
        return (0, 0)   # Convention: 0 = 0*10^0
    mantissa = num/10**exponent

    return (mantissa, int(exponent))

# ***************************************************************************************

def mant_exp_to_num( mant_exp ):
    """
    This function returns a float built with the given (base 10) mantissa and exponent.

    :param mant_exp: (mantissa, exponent) a tuple of two :class:`int` with the mantissa and the exponent of the input number.
    :type mant_exp: tuple
    :return: output number built as mantissa*10**exponent.
    :rtype: :class:`float`

    """
    return mant_exp[0]*10**mant_exp[1]

# ***************************************************************************************

def nice_number( num, mode=0 ):
    """
    This function returns a nice number built with num. This is useful to build the axes of a plot.
    The nice number is built by taking the first digit of the number.

    :param num: input number
    :type num: :class:`float` or :class:`int`
    :param mode: (optional) operation to use to build the nice number

            | 0 -- use ceil
            | 1 -- use round
            | 2 -- use floor

    :type mode: :class:`int`
    :return: a nice number!
    :rtype: :class:`float`

    """
    # extract the mantissa
    exponent = num_to_mant_exp( num )[1]
    # select the working mode
    if ( mode==0 ):
        mantissa = np.ceil( num_to_mant_exp( num )[0])
    elif ( mode==1 ):
        mantissa = np.round( num_to_mant_exp( num )[0])
    elif ( mode==2 ):
        mantissa = np.floor( num_to_mant_exp( num )[0])
    else:
        raise ValueError( 'Wrong worging mode for Fisher_utilities.nice_number' )

    return mant_exp_to_num( ( mantissa, exponent ) )

v_nice_number = np.vectorize(nice_number)

# ***************************************************************************************

def significant_digits( num_err, mode=0 ):
    """
    This function returns the number in num_err at the precision of error.

    :param num_err: (number, error) input number and error in a tuple.
    :type num_err: tuple
    :param mode: (optional) operation to use to build the number

            | 0 -- use ceil
            | 1 -- use round
            | 2 -- use floor

    :type mode: :class:`int`
    :return: a number with all the significant digits according to error
    :rtype: :class:`float`

    """
    number = num_err[0]
    error  = num_err[1]
    number_mant_exp = num_to_mant_exp(number)
    error_mant_exp  = num_to_mant_exp(error)

    temp = mant_exp_to_num( (number_mant_exp[0], number_mant_exp[1]-error_mant_exp[1]) )
    # select the working mode
    if ( mode==0 ):
        temp = np.ceil( temp )
    elif ( mode==1 ):
        temp = np.round( temp )
    elif ( mode==2 ):
        temp = np.floor( temp )
    else:
        raise ValueError('Fisher_utilities.significant_digits called with mode='+str(mode)+' legal values are 0,1,2')

    return temp*10**(error_mant_exp[1])

# ***************************************************************************************

def confidence_coefficient( confidence_level ):
    """
    This function returns the number of sigmas given a confidence level.

    :param confidence_level: desired confidence level. Between 0 and 1.
    :type confidence_level: :class:`float`
    :return: the coefficient (number of sigmas) for the desired confidence level.
    :rtype: :class:`float`

    """
    return np.sqrt(2.)*sp.erfinv(confidence_level)

# ***************************************************************************************

def print_table(table):
    """
    This function prints on the screen a nicely formatted table.

    :param table: a 2D list that should be printed on the screen.

    """
    # transpose the table:
    table = list(map(list, list(zip(*table))))
    # get the column width:
    col_width = [max(len(str(x)) for x in col) for col in zip(*table)]
    # print it to screen:
    print()
    for line in table:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print()

# ***************************************************************************************

def make_list( elements ):
    """
    Checks if elements is a list.
    If yes returns elements without modifying it.
    If not creates and return a list with elements inside.

    :param elements: an element or a list of elements
    :return: a list containing elements if elements is not a list, elements otherwise.
    :rtype: list

    """
    if isinstance(elements, (list, tuple)):
        return elements
    else:
        return [elements]

# ***************************************************************************************

def grouper( n, iterable, fillvalue=None ):
    """
    This small function regroups a list in sub lists of n elements

    :param n: an element or a list of elements
    :param iterable: input list
    :param fillvalue: value to put to fill if no element is present
    :return: a list of list containing grouped elements
    :rtype: list

    """
    args = [iter(iterable)]*n
    return list( it.zip_longest(fillvalue=fillvalue, *args) )

# ***************************************************************************************

def CosmicFish_write_header(name):
    """
    This function prints to screen the CosmicFish header.
    To be called at the beginning of the applications.

    :param name: string that contains the name of the program. This will be printed
        along the CosmicFish header.

    """

    print()
    print("**************************************************************")
    print("   _____               _     _____     __  ")
    print("  / ___/__  ___ __ _  (_)___/ __(_)__ / /  ")
    print(" / /__/ _ \(_-</  ' \/ / __/ _// (_-</ _ \ ")
    print(" \___/\___/___/_/_/_/_/\__/_/ /_/___/_//_/ Py Lib")
    print(" ")
    print("**************************************************************")
    print(name)
    print(" This application was developed using the CosmicFish code.")
    print("**************************************************************")
    print()

# ***************************************************************************************




def _log_add(logx, logy):
    """Add two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
    """Subtract two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x):
    """Compute log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + sp.log_ndtr(-x * 2**0.5)
    except NameError:
        # If log_ndtr is not available, approximate as follows:
        r = sp.erfc(x)
        if r == 0.0:
            # Using the Laurent series at infinity for the tail of the erfc function:
            #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
            # To verify in Mathematica:
            #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
            return (
                -math.log(math.pi) / 2
                - math.log(x)
                - x**2
                - 0.5 * x**-2
                + 0.625 * x**-4
                - 37.0 / 24.0 * x**-6
                + 353.0 / 64.0 * x**-8
            )
        else:
            return math.log(r)


def _log_coef_i(alpha, el, sampling_rate):
    return (
        np.log(sp.binom(alpha, el))
        + (alpha - el) * np.log(1 - sampling_rate)
        + el * np.log(sampling_rate)
    )


def _log_t(log_coef, x, y, sampling_rate):
    return log_coef + x * np.log(sampling_rate) + y * np.log(1 - sampling_rate)


def _log_e(x, y, sigma):
    return np.log(0.5) + _log_erfc((x - y) / (math.sqrt(2) * sigma))


def _log_s(x, log_t, log_e, sigma):
    return log_t + (x * x - x) / (2 * (sigma**2)) + log_e