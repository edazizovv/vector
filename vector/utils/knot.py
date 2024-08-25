#


#
import numpy
from scipy.stats._stats import _kendall_dis
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


#


#
def sd_metric(x, y):
    """
    Somers' D metric.

    Defined with discrete x of arbitrary order of n and y of arbitrary order of m where n may not be equal to m.

    A general case of Gini G1 coefficient and Accuracy Ratio (in the context of credit risk). For more details, see:
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
    https://en.wikipedia.org/wiki/Somers%27_D#Somers'_D_for_binary_dependent_variables

    A link to fast implementation used:
    https://stackoverflow.com/questions/59442544/is-there-an-efficient-python-implementation-for-somersd-for-ungrouped-variables

    :param x: predictor
    :param y: target
    :return: metered value
    """

    assert x.size == y.size

    def count_rank_tie(ranks):
        b_cnt = numpy.bincount(ranks).astype('int64', copy=False)
        b_cnt = b_cnt[b_cnt > 1]
        resulted = ((b_cnt * (b_cnt - 1) // 2).sum(),
                    (b_cnt * (b_cnt - 1.) * (b_cnt - 2)).sum(),
                    (b_cnt * (b_cnt - 1.) * (2 * b_cnt + 5)).sum())
        return resulted

    size = x.size
    perm = numpy.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = numpy.r_[True, y[1:] != y[:-1]].cumsum(dtype=numpy.intp)

    # stable sort on x and convert x to dense ranks
    perm = numpy.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = numpy.r_[True, x[1:] != x[:-1]].cumsum(dtype=numpy.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = numpy.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = numpy.diff(numpy.where(obs)[0]).astype('int64', copy=False)

    n_tie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    x_tie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    y_tie, y0, y1 = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2

    # Note that tot = con + dis + (x_tie - ntie) + (y_tie - n_tie) + n_tie
    #               = con + dis + x_tie + y_tie - n_tie
    # con_minus_dis = tot - x_tie - y_tie + n_tie - 2 * dis

    result = (tot - x_tie - y_tie + n_tie - 2 * dis) / (tot - n_tie)
    return result


def r2_metric(x, y):
    """
    R2 metric.

    Defined with the rate of explained variance, thus may produce negative values.

    For standard definition see:
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions

    :param x: predictor
    :param y: target
    :return: metered value
    """

    assert x.size == y.size

    resulted = r2_score(y_pred=x, y_true=y)

    return resulted


def sp_metric(x, y, method='mean'):
    """
    SMAPE metric.

    (:=Symmetric mean absolute percentage error.)

    Standard definition, see:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    :param x: predictor
    :param y: target
    :param method: can be changed to median to calculate Symmetric median absolute percentage error
    :return: metered value
    """

    assert x.shape == y.shape

    if method == 'mean':
        resulted = numpy.mean((numpy.abs(x - y) / ((numpy.abs(x) + numpy.abs(y)) / 2)))
    elif method == 'median':
        resulted = numpy.median((numpy.abs(x - y) / ((numpy.abs(x) + numpy.abs(y)) / 2)))
    else:
        raise ValueError("Expected 'method' to be either 'mean' or 'median'; received '{0}'".format(method))

    return resulted


def aa_metric(x, y):
    """
    Alpha metric.

    A systematic bias present in x (equal to expected residuals).

    Derived from scikit-learn's implementation of LinearModel estimated with OLS, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    :param x: predictor
    :param y: target
    :return: metered value
    """

    assert x.shape == y.shape

    model_kwg = {}
    model = LinearRegression(**model_kwg)

    model.fit(X=x.reshape(-1, 1), y=y)

    resulted = model.intercept_

    return resulted


def bb_metric(x, y):
    """
    Beta metric.

    A systematic trend present in x.

    Derived from scikit-learn's implementation of LinearModel estimated with OLS, see:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    :param x: predictor
    :param y: target
    :return: metered value
    """

    assert x.shape == y.shape
    model_kwg = {}
    model = LinearRegression(**model_kwg)

    model.fit(X=x.reshape(-1, 1), y=y)

    resulted = model.coef_[0]

    return resulted
