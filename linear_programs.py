import os
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
import matplotlib.pyplot as plt
from itertools import chain, combinations

from ortools.algorithms import pywrapknapsack_solver

# TODO more vectorization for better performance


# helper function
def powerset(iterable):
    """ Calcualtes the powerset, copied from https://docs.python.org/3/library/itertools.html#itertools-recipes """
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate(dim_x, dim_b, random_seed=0, num_x=0, vary_c=False, vary_a=False, vary_b=False, solve=False, problem=None):
    """ Generate one or multiple linear problems (multiple if vary_b or vary_a is True) and num_x instances with an
    about even ratio of being within the LP polytope and not being within it. Some problems with a specific semantic
    meaning can also be generated. """
    if problem is None:  # problem without any particular semantic analogy
        if vary_c is False and vary_a is False and vary_b is False and dim_x == 2:
            # in this scenario, the following data generation functions works very well
            return generate_simple_lp_v2(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve)
        else:
            # otherwise, you this data generation function function which should work well enough
            # there might be some unwanted dependencies though because data is generated in a way which makes the
            # resulting data set balanced (for the (non) feasible class)
            return generate_simple_lp(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve)
    elif problem == "shortest-path":  # shortest path problems
        return generate_shortest_path(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve)


def get_toy_problem(identifier, num_x=0, random_seed=0, **kwargs):
    """ Return a pre-defined LP with a fixed c, a, and b. Always also return the solution because the extra time effort
    is negligible and it can just not be used. """
    identifier = toy_problem_ident(identifier)
    if identifier == "shortest-path-1":
        return toy_sp1(num_x, random_seed)
    if identifier == "shortest-path-2":
        return toy_sp2(num_x, random_seed)
    if identifier == "shortest-path-3":
        return toy_sp3(num_x, random_seed)
    if identifier == "shortest-path-4":
        return toy_sp4(num_x, random_seed)
    if identifier == "shortest-path-5":
        if "ignore_warnings" in kwargs:
            return toy_sp5(num_x, random_seed, ignore_warnings=kwargs["ignore_warnings"])
        return toy_sp5(num_x, random_seed)
    else:
        raise ValueError(f"There is no toy problem for {identifier}")


def toy_problem_ident(identifier):
    """ This function makes it easier to call the toy problems, as they can now be identified by their full name, an
    abbreviation or a simple integer value. """
    if identifier in ["shortest-path-1", "sp1", "0"]:
        return "shortest-path-1"
    if identifier in ["shortest-path-2", "sp2", "1"]:
        return "shortest-path-2"
    if identifier in ["shortest-path-3", "sp3", "2"]:
        return "shortest-path-3"
    if identifier in ["shortest-path-4", "sp4", "3"]:
        return "shortest-path-4"
    if identifier in ["shortest-path-5", "sp5", "4"]:
        return "shortest-path-5"
    if identifier in ["knapsack"]:
        return "knapsack"
    else:
        raise ValueError(f"There is no toy problem for {identifier}")


def generate_simple_lp_v2(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ The goal of this function is to generate b independently from x, and only x dependent on A and b, so that the
    space of b can really be explored. This is done by generating c, A, and b first and independently and then
    generating x in such a way that the complete space of valid instances is part of the generation, the invalid
    instances are generated not too far from the decision boundary (the generated area is a rectangle) and the class
    ratio is about even. The current implementation does not support varying A or b. Unfortunately, the approach in
    this function, while it does work very well for 2 dimension (dim_x==2) does not generalize to larger dimensions.
    Data generation might still possible, but not with the same nice properties (like the whole space of valid instances
    being part of the data generation). It might not work at all for very large dimensions.
    It is recommended to use this function for dim_x==2, dim_b=False, and dim_a=False. Otherwise, it might not work very
    well or not work properly at all. """
    np.random.seed(random_seed)
    # chosen_cost_exception: see the print below
    chosen_cost_exception = dim_x == 2 and dim_b == 3 and not vary_a and not vary_b and not vary_c and random_seed == 7
    if chosen_cost_exception:
        print("For this case, the cost vector is chosen by hand. This should not mean any problem, because in theory "
              "this cost vector could have been randomly generated like that anyway. But since I like to use this seed "
              "for its constraints and its nice, understandable visualization, I also wanted to change the cost vector "
              "in a way which results in the optimal solution lying on the constraint intersection.")

    if num_x == 0:
        # if only the LP needs to be generated
        c = np.random.rand(dim_x)
        if chosen_cost_exception:
            c = np.array([0.5, 0.6])
        a = np.random.rand(dim_b, dim_x)
        # use dim_x as the upper bound for b
        b = np.random.rand(dim_b) * dim_x
        sol = 0
        if solve:
            sol = solve_lp(c, a, b)
        return c, a, b, np.zeros((1, dim_x)), np.array([[0]]), sol
    elif num_x < 10:
        raise ValueError("At least 10 instances need to be generated for some algorithms in this function to work "
                         "properly")

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.rand(dim_x)
        if chosen_cost_exception:
            c = np.array([0.5, 0.6])
    else:
        # one c for each instance
        c = np.random.rand(num_x, dim_x)

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.rand(dim_b, dim_x)
        if not vary_b:
            # one b for all generated instances
            # use dim_x as the upper bound for b
            b = np.random.rand(dim_b) * dim_x

            # calculate the maximum value of each dimension which an instance could have and still be part of a valid
            # solution for the lp (for example if all other elements were 0)
            # if an element was bigger than that max value for the respective dimension, we would already know that
            # Ax <= b could not be true any more
            max_x_values = np.zeros(dim_x)
            # go over each constraint (one row in A and the corresponding element in b)
            for i in range(dim_b):
                max_values_this_dim = b[i] / a[i]
                if i == 0:
                    max_x_values = max_values_this_dim
                else:
                    # element wise minimum
                    max_x_values = np.minimum(max_x_values, max_values_this_dim)

            # now, generate values uniformly within the orthotope (hyperrectangle): this is will either always generate
            # more instances which are within the LP polytope or (on average) just as many instances within as without
            # the LP polytope (this is the case if the LP has only one constraint which is actually relevant, so for
            # example if dim_b=1 or if all but one constraint are always not violated if that one relevant constraint
            # is not violated)

            x = np.random.rand(num_x, dim_x) * max_x_values
            y = label_x_feasible(x, a, b)

            # I decided, that having a class balance with at least 40% and the most 60% in favor of any of the two
            # classes is sufficient but otherwise, x is changed to make the classes more balanced
            # if the class ratio is not within that interval, the data is multiplied by some factor; this should make
            # sense since larger values can be useful to learn the task but values smaller than 0 (negative values) are
            # invalid inputs anyway, so we do not want to get those and "stretching" (multiplying) should make sense

            # it is not really easy to just calculate the "right" factor for the multiplication (for example, imagine,
            # albeit very unlikely, that all generated x are the same; then, no factor will be able to result in a
            # balanced data set; similar problems can also occur for more likely scenarios)
            # the best way I came up with, is using the current percentage as a guideline, for example if there are
            # currently 25% invalid instances, this number should be doubled, so we multiply by 2
            # so we use 0.5 (desired percentage) divided by current percentage of invalid instances

            # trying this nr_tries times, if that still does not result in data within the desired class ration, the
            # last data is still used but a warning is printed
            nr_tries = 50
            for i in range(nr_tries):
                # use a boundary, so that the correction factor does not become too large and we avoid dividing by 0
                # the boundary changes in a way which makes more extremer correction factors impossible for larger
                # iterations (a more "conservative" factor for later iterations)
                current_ratio = min(max(1-np.mean(y), 0.4*((i+1)/nr_tries)), 1-0.4*((i+1)/nr_tries))
                correction_factor = 0.5 / current_ratio
                x = x * correction_factor
                y = label_x_feasible(x, a, b)
                # print(i, np.mean(y))  # if you want to see how the class ratio changes
                if 0.4 <= np.mean(y) <= 0.6:
                    break
                if i == (nr_tries-1):
                    raise Warning("The data for the (non-) feasible task is not well balanced (class ratio:"
                                  f" {np.mean(y)} instances within the LP polytope")

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_lp(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x))
                    for i in range(num_x):
                        sol[i] = solve_lp(c[i], a, b)
        else:
            raise NotImplementedError("Not yet implemented")
    else:
        raise NotImplementedError("Not yet implemented")

    return c, a, b, x, y, sol


def generate_simple_lp_v3(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ The goal of this function is to take generate_simple_lp_v2 and change it in such a way that it also works for
    higher dimensions (dim_x>2). The idea here is, that we start with the same rectangle (orthotope). Now, this
    orthotope can have too few valid instances. So, reducing this orthotope might lead to a balanced dataset, but also
    results in a dataset which does not capture the complete valid space anymore. The idea of this function is to take
    this same orthotope, then generate the data there and sort the data by their summed size (each orthotope dimension
    normalized to 1). Now, by removing the largest ones first, you always start with removing invalid instances far away
    from the decision boundary (so less informative instances) and keeping the whole space of valid instances. The main
    problem of this approach is that you need to generate more data again this way. For example, if you have to delete
    half the data for a balanced dataset (so the original dataset had 25% valid instances vs 75% invalid, removing 50%
    invalid instances leads to a balanced dataset), you have to generate twice as much data (and remove on both
    generated parts). This becomes a big problem for very large dimension where there are close to none valid instances
    within the orthotope.
    I do like the idea behind this function and it really might be the right choice in the correct situation but as of
    now, this function is unused. """
    np.random.seed(random_seed)

    if num_x == 0:
        # if only the LP needs to be generated
        c = np.random.rand(dim_x)
        a = np.random.rand(dim_b, dim_x)
        # use dim_x as the upper bound for b
        b = np.random.rand(dim_b) * dim_x
        sol = 0
        if solve:
            sol = solve_lp(c, a, b)
        return c, a, b, np.zeros((1, dim_x)), np.array([[0]]), sol
    elif num_x < 10:
        raise ValueError("At least 10 instances need to be generated for some algorithms in this function to work "
                         "properly")

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.rand(dim_x)
    else:
        # one c for each instance
        c = np.random.rand(num_x, dim_x)

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.rand(dim_b, dim_x)
        if not vary_b:
            # one b for all generated instances
            # use dim_x as the upper bound for b
            b = np.random.rand(dim_b) * dim_x

            # calculate the maximum value of each dimension which an instance could have and still be part of a valid
            # solution for the lp (for example if all other elements were 0)
            # if an element was bigger than that max value for the respective dimension, we would already know that
            # Ax <= b could not be true any more
            max_x_values = np.zeros(dim_x)
            # go over each constraint (one row in A and the corresponding element in b)
            for i in range(dim_b):
                max_values_this_dim = b[i] / a[i]
                if i == 0:
                    max_x_values = max_values_this_dim
                else:
                    # element wise minimum
                    max_x_values = np.minimum(max_x_values, max_values_this_dim)

            # now, generate values uniformly within the orthotope (hyperrectangle)

            x = np.random.rand(num_x, dim_x) * max_x_values
            y = label_x_feasible(x, a, b)

            # I decided, that having a class balance with at least 40% and the most 60% in favor of any of the two
            # classes is sufficient but otherwise, x is changed to make the classes more balanced
            # if the class ratio is too unbalanced because of too many feasible instances, the data is multiplied by
            # some factor; this should make sense since larger values can be useful to learn the task but values smaller
            # than 0 (negative values) are invalid inputs anyway, so we do not want to get those and "stretching"
            # (multiplying) should make sense
            # this does not guarantee that a sufficient class balance will be reached but it makes it likely enough and
            # works well in practice
            upper_factor = 2
            lower_factor = 1
            multiply_data = np.mean(y) > 0.6
            while multiply_data:
                factor = (upper_factor + lower_factor) / 2
                x_new = x * factor
                y = label_x_feasible(x_new, a, b)
                if np.mean(y) > 0.6:
                    lower_factor = factor
                elif np.mean(y) < 0.4:
                    upper_factor = factor
                else:
                    x = x_new
                    multiply_data = False

            if np.mean(y) < 0.4:
                # remove the largest data points

                # calculate how much data you will need to, after removing data, end up with num_x data points
                more_data_factor = 0.5/np.mean(y)
                more_data = num_x * (more_data_factor-1)
                # generate more data
                x_more = np.random.rand(int(more_data), dim_x) * max_x_values
                x_all = np.concatenate((x, x_more))
                # sort data by the sum of their (normalized) dimension values
                indices = np.argsort(np.sum(x_all / max_x_values, axis=1))
                x_new = x_all[indices]
                # remember number of valid instances before removing
                old_valids = np.sum(label_x_feasible(x_new, a, b))
                # remove large instances
                x_new = x_new[:num_x]
                y = label_x_feasible(x_new, a, b)
                new_valids = np.sum(y)
                if old_valids != new_valids:
                    # I think (but without proof and I am also not 100% sure) that this way of removing instances should
                    # never remove valid instances. However, if it does, then this approach might be a bad idea and I
                    # would have to look more closely into it.
                    # This condition above these comments exists to trigger if my assumption is false.
                    raise ValueError("During the data generation, unexpected behavior occurred. Check the code please.")
                if 0.4 <= np.mean(y) <= 0.6:
                    x = x_new
                    np.random.shuffle(x)  # TODO could shuffle indices and keep labels y for more efficiency
                    y = label_x_feasible(x, a, b)
                else:
                    # this could either happen for not enough data (randomness) or because my code does not behave as
                    # intended
                    # TODO think about this and the error above, can I deal with this better? loop and try again?
                    raise ValueError("During the data generation, unexpected behavior occurred. Check the code please.")

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_lp(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x))
                    for i in range(num_x):
                        sol[i] = solve_lp(c[i], a, b)
        else:
            raise NotImplementedError("Not yet implemented")
    else:
        raise NotImplementedError("Not yet implemented")

    return c, a, b, x, y, sol


def generate_simple_lp(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ Generate one or multiple linear problems (multiple if vary_b or vary_a is True) and num_x instances with an
    about even ratio of being within the LP polytope and not being within it. Results are either one-dimensional arrays
    of the respective length or have the shape (num_x, dim) with the respective dimension length 'dim'.
    Some performance improvements are certainly possible but at the time of writing not worth the extra effort. """
    np.random.seed(random_seed)

    if num_x == 0:
        # if only the LP needs to be generated
        c = np.random.rand(dim_x)
        a = np.random.rand(dim_b, dim_x)
        # by applying the following multiplication with 0.5 dim_x, it is ensured that on average Ax is about as large as
        # b if instances x would also be generated with values randomly within [0, 1]
        b = np.random.rand(dim_b) * (1 / 2 * dim_x)
        sol = 0
        if solve:
            sol = solve_lp(c, a, b)
        return c, a, b, np.zeros((1, dim_x)), np.array([[0]]), sol

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.rand(dim_x)
    else:
        # one c for each instance
        c = np.random.rand(num_x, dim_x)

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.rand(dim_b, dim_x)
        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.rand(num_x, dim_x)

            # if for every comparison of dim_b ((Ax)[i] <= b[i] for all i in [1,...,dim_b]) the following probability
            # describes the probability of each of these comparisons being true, then the overall probability of
            # Ax <= b will be true in about 50% of all cases, this is used to try to generate instances which are
            # relatively balanced in terms of satisfying Ax <= b or not
            prob_per_b = 0.5 ** (1 / dim_b)

            # set b so that for every dimension prob_per_b instances result in smaller values
            b = np.zeros(dim_b)
            ax = np.matmul(a, x.T)  # dimensions: dim_b * num_x
            rest_ax = np.copy(ax)
            for i in range(b.shape[0]):
                b[i] = np.quantile(rest_ax[i], prob_per_b)
                rest_ax = rest_ax.T[rest_ax[i] <= b[i]]
                rest_ax = rest_ax.T

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_lp(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x))
                    for i in range(num_x):
                        sol[i] = solve_lp(c[i], a, b)
        else:
            # one b for each instance

            # generate x
            x = np.random.rand(num_x, dim_x)

            # set b so that Ax <= b in about half of the cases
            b = np.zeros((num_x, dim_b))
            for i in range(num_x):
                if np.random.choice([0, 1]):
                    # 50% of the time, just set b randomly, in the same interval as x
                    # especially for higher dimensions, this will mostly (but not always) results in Ax > b
                    b[i] = np.random.rand(dim_b)
                else:
                    # 50% of the time, set b so that it satisfies Ax <= b
                    # here, b is set randomly in the interval between Ax and a maximum value
                    # the maximum value is the largest possible result of Ax when A and x are both generated as values
                    # between 0 and 1 (which is the case)
                    ax = np.matmul(a, x[i].reshape((dim_x, 1)))
                    temp = np.full_like(ax, dim_x) - ax
                    b[i] = (ax + np.random.rand(dim_b).reshape((dim_b, 1)) * temp)[:, 0]

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a, b[i])
                    else:
                        sol[i] = solve_lp(c[i], a, b[i])
    else:
        # one a for each instance
        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.rand(num_x, dim_x)

            # generate a
            a = np.random.rand(num_x, dim_b, dim_x)

            # if for every comparison of dim_b ((Ax)[i] <= b[i] for all i in [1,...,dim_b]) the following probability
            # describes the probability of each of these comparisons being true, then the overall probability of
            # Ax <= b will be true in about 50% of all cases, this is used to try to generate instances which are
            # relatively balanced in terms of satisfying Ax <= b or not
            prob_per_b = 0.5 ** (1 / dim_b)

            # set b so that for every dimension prob_per_b instances result in smaller values
            b = np.zeros(dim_b)
            ax = np.zeros((dim_b, num_x))  # dimensions: dim_b * num_x
            for i in range(num_x):
                ax[:, i] = np.matmul(a[i], x[i].T)
            rest_ax = np.copy(ax)
            for i in range(b.shape[0]):
                b[i] = np.quantile(rest_ax[i], prob_per_b)
                rest_ax = rest_ax.T[rest_ax[i] <= b[i]]
                rest_ax = rest_ax.T

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a[i], b)
                    else:
                        sol[i] = solve_lp(c[i], a[i], b)
        else:
            # one b for each instance

            # generate a
            a = np.random.rand(num_x, dim_b, dim_x)

            # generate x
            x = np.random.rand(num_x, dim_x)

            # set b so that Ax <= b in about half of the cases
            b = np.zeros((num_x, dim_b))
            for i in range(num_x):
                if np.random.choice([0, 1]):
                    # 50% of the time, just set b randomly, in the same interval as x
                    # especially for higher dimensions, this will mostly (but not always) results in Ax > b
                    b[i] = np.random.rand(dim_b)
                else:
                    # 50% of the time, set b so that it satisfies Ax <= b
                    # here, b is set randomly in the interval between Ax and a maximum value
                    # the maximum value is the largest possible result of Ax when A and x are both generated as values
                    # between 0 and 1 (which is the case)
                    ax = np.matmul(a[i], x[i].reshape((dim_x, 1)))
                    temp = np.full_like(ax, dim_x) - ax
                    b[i] = (ax + np.random.rand(dim_b).reshape((dim_b, 1)) * temp)[:, 0]

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x))
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_lp(c, a[i], b[i])
                    else:
                        sol[i] = solve_lp(c[i], a[i], b[i])

    y = label_x_feasible(x, a, b)

    return c, a, b, x, y, sol


def label_x_feasible(x, a, b):
    """ For all instances x, return a label describing whether x is a possible solution (not an optimal one) for the LP
    specified by a and b. Therefore, the label is 1 if x >= 0 and Ax <= b and 0 otherwise. """
    y = np.zeros((x.shape[0], 1))
    if len(a.shape) == 2 and len(b.shape) == 1:
        # fixed A and b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a, x[i]) <= b):
                y[i] = 1
            else:
                y[i] = 0
    elif len(a.shape) == 2 and len(b.shape) == 2:
        # fixed A, varying b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a, x[i]) <= b[i]):
                y[i] = 1
            else:
                y[i] = 0
    elif len(a.shape) == 3 and len(b.shape) == 1:
        # varying A, fixed b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a[i], x[i]) <= b):
                y[i] = 1
            else:
                y[i] = 0
    else:
        # varying A and b
        for i in range(x.shape[0]):
            if np.all(x[i] >= 0) and np.all(np.matmul(a[i], x[i]) <= b[i]):
                y[i] = 1
            else:
                y[i] = 0
    return y


def solve_lp(c, a, b, method="interior-point"):
    """ Solve a given LP using scipy.linprog. Keep in mind that here, we maximize the cost (gain). """
    # I found out that interior point does not always perform well, for example for shortest path problems
    res = linprog(-c, a, b, method=method)
    return res.x


def generate_x(c, a, b, num_x, random_seed):
    """ Given a LP (c, a, b), generate num_x number of instances. I have not yet found a good way to generate such
    instances in a relatively balanced way for LPs with different dimension. """

    raise NotImplementedError("No functioning implementation finished yet.")
    # # one possible idea to generate instances in a relatively balanced way (as many valid as invalid instances)
    #
    # np.random.seed(random_seed)
    #
    # # get the optimal solution
    # optimal_x = solve_lp(c, a, b)
    #
    # # use twice the optimal solution (of course invalid) as upper bound for randomly generated instances
    # bounds = 0.02 * np.max(optimal_x)
    #
    # # generate instances in with values between 0 and the respective bounds value
    # all_x = np.zeros((num_x, a.shape[1]))
    # all_y = np.zeros((num_x, 1))
    # for i in range(num_x):
    #     all_x[i] = np.random.rand(len(c)) * bounds
    #     all_y[i] = int(np.all(np.matmul(a, all_x[i]) <= b))
    # return all_x, all_y


def generate_shortest_path(dim_x, dim_b, random_seed, num_x, vary_c, vary_a, vary_b, solve):
    """ Generate one or multiple (multiple if vary_b or vary_a is True) linear problems representing shortest path
    problems and num_x instances with an about even ratio of being within the LP polytope and not being within it.
    Results are either one-dimensional arrays of the respective length or have the shape (num_x, dim) with the
    respective dimension length 'dim'. """
    raise NotImplementedError("Not implemented yet.")


def toy_sp1(num_x, random_seed):
    """ Return a rather small shortest path problem. The graph used here is taken from
    https://en.wikipedia.org/wiki/Shortest_path_problem (accessed 16.07.2021). """
    # TODO go over the x generation again, can maybe improve that (maybe after implementing generate_x or generate_shortest_path
    c = np.array([4, 2, 5, 10, 3, 11, 4]) * -1  # negative because we maximize costs (gain)
    b = np.array([1, 0, 0, 0, 0, -1])
    a = np.array([[1, 1, 0, 0, 0, 0, 0],
                  [-1, 0, 1, 1, 0, 0, 0],
                  [0, -1, -1, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0, 1, -1],
                  [0, 0, 0, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, -1, 0]])

    if num_x == 0:
        return c, a, b

    print("Does not return balanced instances (much more invalid solutions but should be fine for most purposes)")
    np.random.seed(random_seed)
    x = np.zeros((num_x, len(c)))
    y = np.zeros((num_x, 1))
    for i in range(num_x):
        x[i] = np.random.choice([0, 1], len(c))
        y[i] = int(np.all(np.matmul(a, x[i]) <= b))

    sol = solve_lp(c, a, b, method="highs-ds")

    return c, a, b, x, y, sol


def toy_sp1b(num_x, random_seed):
    """ A variation of toy_sp1, but much larger. Has very few valid paths (in comparison to invalid paths). One
    important downside of this toy problem is that there are very few valid paths (the data is very unbalanced), which
    makes learning with a NN quite difficult. Currently this function is unused. """
    c = np.ones(15) * -1  # cost are irrelevant anyway for single lp
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, -1])
    a = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, -1, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -1, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1]])

    # just so that the default seed (0) returns half valid paths in first half of the data (train and test set)
    np.random.seed(random_seed+2)

    if num_x == 0:
        return c, a, b
    else:
        print(f"Changing \"num_x\" to {2**15} (2^15) to get each path once for this toy problem.")
        num_x = 2**15

    x = np.zeros((num_x, len(c)))
    y = np.zeros((num_x, 1))
    for i in range(2**15):
        binary_str = (bin(i).split("b")[1])[::-1]  # reverse, so that digits go from small to high
        for j in range(len(binary_str)):
            if binary_str[j] == "1":
                x[i, j] = 1

    np.random.shuffle(x)

    for i in range(2 ** 15):
        y[i] = int(np.all(np.matmul(a, x[i]) <= b))

    print("There are barely valid paths, which might lead to problems for learning.")

    sol = solve_lp(c, a, b, method="highs-ds")

    return c, a, b, x, y, sol


def toy_sp2(num_x, random_seed):
    """ Using a predefined cost vector and 6 predefined edges which make sure that a solution always exists, create a
    shortest path problem with 12 edges (6 generated ones) per num_x and one instance x each. """
    # CAUTION: dim_b and dim_x are NOT parameters, you would have to change more than these values for this function to
    # work as intended
    dim_b = 6
    dim_x = 12
    c = np.array([5, 5, 5, 5, 5, 5, 1, 2, 3, 4, 5, 6]) * -1  # negative because we maximize costs (gain)
    b = np.array([1, 0, 0, 0, 0, -1])

    # fixed edges: A->B, B->C, C->D, D->E, E->F, F->A
    a_base = np.array([[1, 0, 0, 0, 0, -1],
                      [-1, 1, 0, 0, 0, 0],
                      [0, -1, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, 0],
                      [0, 0, 0, -1, 1, 0],
                      [0, 0, 0, 0, -1, 1]])

    # now every nodes get an additional edge
    a = np.zeros((num_x, dim_b, dim_x))
    x = np.zeros((num_x, dim_x))
    y = np.zeros((num_x, 1))
    sol = np.zeros((num_x, dim_x))

    np.random.seed(random_seed)

    print("The x generation might be suboptimal. This might be a possible point of improvement if results using this "
          "toy example (shortest-path-2) disappoint")
    # referring to the print above: at the time of writing this comment, my model achieves an accuracy of 99% for
    # (not) feasible path classification. What if that is not really learned but it is learned that the A-B-C-D-E-F path
    # is valid and that the shortest path (which I really think is learned) is valid. Maybe the 1% missing are those
    # instances which are valid paths which do not fall in those two categories. If this is true, this problem might
    # be a result of the bad x generation

    for i in range(num_x):
        # generate a
        a_var = np.zeros((dim_b, dim_x // 2))
        edge_from = np.random.randint(0, dim_b, size=(dim_x // 2))
        # the two offset here and below: this way we avoid, that an edge would go to the same node it came from (we
        # could not model this because the value can only be 1 or -1); we skip the next position as well because we
        # don't want a node to go to the next node, as we already have those node in a_base
        edge_to = np.random.randint(0, dim_b-2, size=(dim_x // 2))
        edges = np.stack((edge_from, edge_to))
        # to avoid having the same edge two times
        while np.unique(edges, axis=1).shape[1] != edges.shape[1]:
            # just do it again, this should happen rarely enough so that this very simple approach should be good enough
            edge_from = np.random.randint(0, dim_b, size=(dim_x // 2))
            edge_to = np.random.randint(0, dim_b - 2, size=(dim_x // 2))
            edges = np.stack((edge_from, edge_to))
        edge_to = (edge_from + edge_to + 2) % dim_b
        a_var[edge_from, list(range(dim_x//2))] = 1
        a_var[edge_to, list(range(dim_x//2))] = -1
        a[i] = np.concatenate((a_base, a_var), axis=1)

        # generate x and set y and sol (set x so that the first 8 edges are always active)
        # x[i] = np.concatenate((x_base, np.random.choice([0, 1], dim_x//2)))
        x[i] = np.random.choice([0, 0, 1], dim_x)  # results y[i] == 1 about 0.5% of the time
        y[i] = int(np.all(np.matmul(a[i], x[i]) <= b))
        sol[i] = solve_lp(c, a[i], b, method="highs-ds")
        # if the randomly generated path is not valid (which is the vast majority)...
        if y[i] == 0:
            random_number = np.random.random(1)[0]
            # ... 10% of the time take the optimal (shortest path) solution (we know this is valid)
            if random_number < 0.1:
                x[i] = sol[i]
            # another 10% of the time use the fixed edges for a solution we know is valid
            elif random_number < 0.2:
                x[i] = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            # calculate y again, technically unnecessary but provides an extra safety net
            y[i] = int(np.all(np.matmul(a[i], x[i]) <= b))

    return c, a, b, x, y, sol


def toy_sp3(num_x, random_seed):
    """ This toy example contains a larger graph than sp2. Here, the variability is created by omitting edges randomly.
    Only large edge guarantees that a solution always exists but this large edge is very expensive. All other edges
    are bidirectional (coded as two directional edges). """
    # CAUTION: dim_b and dim_x are NOT parameters, you would have to change more than these values for this function to
    # work as intended
    dim_b = 14
    dim_x = 59
    # one row per start node (technically omnidirectional)
    edge_costs = np.array([15, 12, 19,
                           9, 19, 12,
                           3, 15, 6, 22,
                           13,
                           7, 15, 29,
                           6, 19,
                           11, 13,
                           5, 14, 19,
                           12, 13, 10,
                           17,
                           17,
                           2, 14,
                           13
                           ])
    edge_costs = np.concatenate((edge_costs, edge_costs))
    c = np.concatenate((edge_costs, np.array([200]))) * -1  # negative because we maximize costs (gain)
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])

    # nodes are A (0) to M (12) like the alphabet with the end node O (13)
    edges = [(0, 1), (0, 2), (0, 3),
             (1, 2), (1, 4), (1, 5),
             (2, 3), (2, 5), (2, 6), (2, 8),
             (3, 6),
             (4, 5), (4, 7), (4, 10),
             (5, 7), (5, 8),
             (6, 8), (6, 9),
             (7, 8), (7, 10), (7, 11),
             (8, 9), (8, 11), (8, 12),
             (9, 12),
             (10, 13),
             (11, 12), (11, 13),
             (12, 13)]
    edges += [(e1, e0) for e0, e1 in edges]
    edges += [(0, 13)]  # guaranteed edge

    a = np.zeros((num_x, dim_b, dim_x))

    # fill all edges of a
    for count, e in enumerate(edges):
        a[:, e[0], count] = 1
        a[:, e[1], count] = -1

    np.random.seed(random_seed)

    # make it a little parametric by removing edges
    keep_edge = 0.75  # probability of keeping any edge
    edge_ind = np.random.choice([0, 1], (num_x, (dim_x-1)//2), p=[1-keep_edge, keep_edge])
    edge_ind = np.concatenate((edge_ind, edge_ind), axis=1)
    indices = np.where(edge_ind == 0)
    a[indices[0], :, indices[1]] = 0

    # optimal path
    sol = np.zeros((num_x, dim_x))
    for i in range(a.shape[0]):
        sol[i] = solve_lp(c, a[i], b, method="highs-ds")

    # I did not find a good way to randomly generate valid paths, therefore the following approach
    # First, just set random edges (does not make much sense, also chooses inactive edges)
    x = np.random.choice([0, 1], size=(num_x, dim_x))
    indices = np.where(np.random.choice([0, 1], num_x) == 1)
    # After that, set the optimal solution about 50% of the time
    x[indices] = sol[indices]
    y = np.zeros((num_x, 1))
    for i in range(a.shape[0]):
        y[i] = int(np.all(np.matmul(a[i], x[i]) <= b))

    # Note: if I want to, I can look at "np.sum(sol[:,i])" to see how often edge i was used and I could use this
    # knowledge to change the graph so that optimal paths are generally more diverse
    return c, a, b, x, y, sol


def toy_sp4(num_x, random_seed):
    """ This shortest path toy problem generates a graph with 6 nodes and 5 predefined edges which make sure that a
    solution always exists. Another 6 variables edges are added for variability. Start and end node are always the same.
    For each different graph (set of variables edges), the edge order in a is shuffled (this does not change the
    problem, but the idea is that this way the network is forced to learn from the entire a matrix and not only from
    the columns containing variable edges and in other shortest path problem setups) and generate a new random cost
    vector. Also calculate and return the optimal solution (shortest path). In line with previous shortest path problem
    generation functions, one input vector (possible LP solution) is created for each graph, together with the
    corresponding label describing whether this is a valid LP solution (1) or not (0). """
    # CAUTION: dim_b and dim_x are NOT parameters, you would have to change more than these values for this function to
    # work as intended
    dim_b = 6
    dim_x = 12
    b = np.array([1, 0, 0, 0, 0, -1])

    # fixed edges: A->B, B->C, C->D, D->E, E->F, F->G
    a_base = np.array([[1, 0, 0, 0, 0, -1],
                      [-1, 1, 0, 0, 0, 0],
                      [0, -1, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, 0],
                      [0, 0, 0, -1, 1, 0],
                      [0, 0, 0, 0, -1, 1]])

    # now every nodes get an additional edge
    a = np.zeros((num_x, dim_b, dim_x))
    x = np.zeros((num_x, dim_x))
    y = np.zeros((num_x, 1))
    sol = np.zeros((num_x, dim_x))
    c = np.zeros((num_x, dim_x))

    np.random.seed(random_seed)

    print("The x generation might be suboptimal. This might be a possible point of improvement if results using this "
          "toy example (shortest-path-4) disappoint")
    # Referring to the print above: look at shortest-path-2 for more information. While most of this (if not all) should
    # apply here as well, this toy problem was designed for a neural network setup in mind where x and y are unused.

    for i in range(num_x):
        # generate a
        a_var = np.zeros((dim_b, dim_x // 2))
        edge_from = np.random.randint(0, dim_b, size=(dim_x // 2))
        # the two offset here and below: this way we avoid, that an edge would go to the same node it came from (we
        # could not model this because the value can only be 1 or -1); we skip the next position as well because we
        # don't want a node to go to the next node, as we already have those node in a_base
        edge_to = np.random.randint(0, dim_b-2, size=(dim_x // 2))
        edges = np.stack((edge_from, edge_to))
        # to avoid having the same edge two times
        while np.unique(edges, axis=1).shape[1] != edges.shape[1]:
            # just do it again, this should happen rarely enough so that this very simple approach should be good enough
            edge_from = np.random.randint(0, dim_b, size=(dim_x // 2))
            edge_to = np.random.randint(0, dim_b - 2, size=(dim_x // 2))
            edges = np.stack((edge_from, edge_to))
        edge_to = (edge_from + edge_to + 2) % dim_b
        a_var[edge_from, list(range(dim_x//2))] = 1
        a_var[edge_to, list(range(dim_x//2))] = -1
        a[i] = np.concatenate((a_base, a_var), axis=1)

        # change edge order of a so that (hopefully) the network can actually learn with all 12 dimensions (edges) of a
        np.random.shuffle(a[i].T)

        # generate c (it might be interesting to try out different cost vector generation, for example this current one
        # can result in very different path length (1 to 10, so up to 10 times higher) while for example low=1, high=16
        # would still result in quite different, but not as extreme path length differences)
        c[i] = np.random.randint(1, high=11, size=dim_x) * -1

        # generate x and set y and sol (set x so that the first 8 edges are always active)
        # again, note that I did not put any effort into generating x for this toy_sp4 problem and only copied it from
        # toy_sp2, so the generation might be decent, it also might be rather bad
        x[i] = np.random.choice([0, 0, 1], dim_x)
        y[i] = int(np.all(np.matmul(a[i], x[i]) <= b))
        sol[i] = solve_lp(c[i], a[i], b, method="highs-ds")
        # if the randomly generated path is not valid (which is the vast majority)...
        if y[i] == 0:
            random_number = np.random.random(1)[0]
            # ... 10% of the time take the optimal (shortest path) solution (we know this is valid)
            if random_number < 0.1:
                x[i] = sol[i]
            # another 10% of the time use the fixed edges for a solution we know is valid
            elif random_number < 0.2:
                x[i] = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
            # calculate y again, technically unnecessary but provides an extra safety net
            y[i] = int(np.all(np.matmul(a[i], x[i]) <= b))

    return c, a, b, x, y, sol


def all_valid_subpaths(a, b):
    """ Given a graph defined by a and b (as in a LP shortest path formulation), return all valid subpaths.
    What is a valid path? Each path from start to goal where each edge in the path is chosen not more than once, this
    means that cycles are possible, however, once the goal node has been reached a path is finished (a path from start
    to some nodes to goal to another node and back to the goal is not valid, in this case the path would be finished
    once it reached the goal for the first time).
    Completely empty as well as finished paths are also considered valid subpaths in this function.
    """
    possible_paths = []
    valid_paths = []
    goal_node = np.where(b == -1)
    # find all valid paths (basically a breadth-first search)
    for i in range(a.shape[1]):
        if i == 0:  # start of paths
            current_node = np.where(b == 1)
            for j in range(a.shape[1]):
                if a[current_node, j] == 1:
                    # path from start to goal
                    if a[goal_node, j] == -1:
                        valid_paths.append([j])
                    # path from start to another node
                    else:
                        possible_paths.append([j])
        else:
            path_iterator = possible_paths
            possible_paths = []
            for path in path_iterator:
                current_node = np.where(a[:, path[-1]] == -1)
                for j in range(a.shape[1]):
                    if a[current_node, j] == 1:
                        # reached goal node
                        if a[goal_node, j] == -1:
                            valid_paths.append(path + [j])
                        # reached another node
                        else:
                            # do not store paths where an edge is chosen more than once
                            if j not in path:
                                possible_paths.append(path + [j])
            if len(possible_paths) == 0:
                break

    # get all possible subpaths (we know that you can make possible solutions (paths) with those)
    # using a set to avoid duplicate subpaths
    all_sub_paths = set()
    for path in valid_paths:
        for sub_path in powerset(path):
            sp = list(sub_path)
            sp.sort()
            all_sub_paths.add(tuple(sp))

    # transform all_sub_paths into x vector form
    valid_x = np.zeros((len(all_sub_paths), a.shape[1]))
    for i, path in enumerate(all_sub_paths):
        for edge in path:
            valid_x[i, edge] = 1

    return valid_x


def reset_data(random_seed):
    """ Since toy_sp5 appends to existing data, if you want a clean new start, the data needs to be deleted and reset
    first. The one row with zeros is removed in toy_sp5 when loading the data again. """
    a = np.zeros((1, 6, 11))
    np.save(f"data/rs_{random_seed}_a.npy", a)
    sol = np.zeros((1, 11))
    np.save(f"data/rs_{random_seed}_sol.npy", sol)
    x = np.zeros((1, 12))
    np.save(f"data/rs_{random_seed}_x.npy", x)
    y = np.zeros((1, 12))
    np.save(f"data/rs_{random_seed}_y.npy", y)


def solve_lp_given_x(c, a, b, x):
    """ Solve the given LP (c, a, b) given that the inputs in x are fixed (only written for shortest path problems)."""
    # TODO known minor bug/problem: If a subpath for which a solution is calculated contains a cycle, this cycle...
    # TODO ...might be isolated in the solution because this would still result in a valid solution. Example: input:...
    # TODO ...x=(B-->C, C-->B), solution: sol=(B-->C, C-->B, A-->F)

    # about the two concatenates: for some subpaths leading or including loops, the optimal solution given this subpath
    # can include choosing a shortest path twice (per the LP formulation); to circumvent that, I add an additional
    # constraint for each edge which restricts each edge to a value <=1 which results in solutions where each edge is
    # either not taken or taken only once
    a_given_x = np.copy(a)
    a_given_x = np.concatenate((a_given_x, np.diag(np.ones(11))), axis=0)
    b_given_x = np.copy(b)
    b_given_x = np.concatenate((b_given_x, np.ones(11)))
    for edge in range(x.shape[0]):
        if x[edge] == 1:
            for row in range(a_given_x.shape[0]):
                if a_given_x[row, edge] == 1:
                    a_given_x[row, edge] = 0
                    b_given_x[row] -= 1
                elif a_given_x[row, edge] == -1:
                    a_given_x[row, edge] = 0
                    b_given_x[row] += 1
    # Sometimes the console print "HQPrimal::solvePhase2"; I took some times looking into that and I am pretty sure that
    # the solution when this is printed is still completely valid, so I think it is just annoying (because of the
    # console output). Still, I don not understand why this happens sometimes.

    # I decided to add the input x to the solution, I think this makes more sense, especially for shortest path
    return solve_lp(c, a_given_x, b_given_x, method="highs-ds") + x


def toy_sp5(num_x, random_seed, ignore_warnings=False):
    """ This shortest path toy problem generates a graph with 6 nodes and 5 predefined edges which make sure that a
    solution always exists. Another 6 variables edges are added for variability. Start and end node are always the same.
    For each different graph (set of variables edges), the edge order in a is shuffled (this does not change the
    problem, but the idea is that this way the network is forced to learn from the entire a matrix and not only from
    the columns containing variable edges and in other shortest path problem setups). The cost vector is generated
    randomly once but is fixed for all graphs for that random seed. Also calculate and return the optimal solution
    (shortest path). Contrary to previous shortest path problem generation functions, x and y DO NOT represent a random
     input and whether this is valid or not. Instead, each x is a random subpath and the corresponding y is the optimal
     solution including this subpath.
     The parameter num_x determines how many different graphs are generated. For each graph, every possible subpath is
     generated (x) and the corresponding solution calculated (y). This results in much more different instances for x
     than a (on average more than one or two hundred subpaths for each graph). """
    # see if data can be loaded (loaded and storing here is implemented because generating can take a significant
    # amount of time)
    try:
        disk_b = np.load(f"data/rs_{random_seed}_b.npy")
        disk_c = np.load(f"data/rs_{random_seed}_c.npy")
        disk_a = np.load(f"data/rs_{random_seed}_a.npy")
        disk_sol = np.load(f"data/rs_{random_seed}_sol.npy")
        disk_x = np.load(f"data/rs_{random_seed}_x.npy")
        disk_y = np.load(f"data/rs_{random_seed}_y.npy")
        nr_a = disk_a.shape[0]
        nr_x = disk_x.shape[0]
        if ignore_warnings:
            y_n = "y"
        else:
            input_str = f"There is data for {nr_a} different graphs with overall {nr_x} different subpaths which can " \
                        f"be loaded. Instead of generating new data, should this data be loaded? (y/n) "
            y_n = input(input_str).lower()
            while y_n not in ["y", "n"]:
                print("Please enter \"y\" for yes or \"n\" for no.")
                y_n = input(input_str).lower()
        # loading the data
        if y_n == "y":
            # shapes which will be returned
            a = np.zeros((disk_x.shape[0], disk_a.shape[1], disk_a.shape[2]))
            sol = np.zeros((disk_x.shape[0], disk_sol.shape[1]))
            x = np.zeros((disk_x.shape[0], disk_x.shape[1] - 1))
            y = np.zeros((disk_y.shape[0], disk_y.shape[1] - 1))

            # the first column in x and y links to the corresponding graph (a), this is done to save storage for a
            # here, create a so that is has as many instances as x and that a, sol, x, and y can be indexed together
            for index in range(disk_x.shape[0]):
                a_index = int(disk_x[index, 0])
                a[index] = disk_a[a_index]
                sol[index] = disk_sol[a_index]
                x[index] = disk_x[index, 1:]
                y[index] = disk_y[index, 1:]

            # shuffle
            np.random.seed(random_seed)
            indices = np.random.permutation(x.shape[0])
            a = a[indices]
            x = x[indices]
            y = y[indices]
            sol = sol[indices]

            return disk_c, a, disk_b, x, y, sol
        else:
            con = input(f"This will overwrite existing data (\"data/rs_{random_seed}_[a,b,c,sol,x,y].npy\"). "
                        f"Confirm that you want to continue. (y/n)")
            if con.lower() != "y":
                raise RuntimeError("Program can not continue without loading or deleting existing data.")
    except FileNotFoundError:
        print("There is no existing data for toy_sp5 with this random seed, generating the data now.")

    reset_data(random_seed)  # TODO implement a way to append to existing data to make generating more easier

    # CAUTION: dim_b and dim_x are NOT parameters, you would have to change more than these values for this function to
    # work as intended
    dim_b = 6
    dim_x = 11
    b = np.array([1, 0, 0, 0, 0, -1])
    # how to define/generate c is something which could also be done differently
    c = np.random.randint(1, high=9, size=dim_x) * -1

    np.save(f"data/rs_{random_seed}_b.npy", b)
    np.save(f"data/rs_{random_seed}_c.npy", c)

    # fixed edges: A->B, B->C, C->D, D->E, E->F
    a_base = np.array([[1, 0, 0, 0, 0],
                      [-1, 1, 0, 0, 0],
                      [0, -1, 1, 0, 0],
                      [0, 0, -1, 1, 0],
                      [0, 0, 0, -1, 1],
                      [0, 0, 0, 0, -1]])

    # more edges will be generated randomly
    # TODO (not necessary, but an idea for a possible improvement for a generation): set other (random, variable)...
    # TODO ... edges more purposefully (not completely random), like maybe only forward edges (e.g. not E-->B) or...
    # TODO ...allow those but always have a cheaper cost here, would have to think about the specifics a bit more
    # about it more but it might be interesting
    a = np.zeros((num_x, dim_b, dim_x))
    sol = np.zeros((num_x, dim_x))

    np.random.seed(random_seed)

    sub_path_count = 0
    for i in range(num_x):
        # generate a
        a_var_nr = 6
        a_var = np.zeros((dim_b, a_var_nr))
        edge_from = np.random.randint(0, dim_b, size=a_var_nr)
        # the two offset here and below: this way we avoid, that an edge would go to the same node it came from (we
        # could not model this because the value can only be 1 or -1); we skip the next position as well because we
        # don't want a node to go to the next node, as we already have those node in a_base
        edge_to = np.random.randint(0, dim_b-2, size=a_var_nr)
        edges = np.stack((edge_from, edge_to))
        # to avoid having the same edge two times
        while np.unique(edges, axis=1).shape[1] != edges.shape[1]:
            # just do it again, this should happen rarely enough so that this very simple approach should be good enough
            edge_from = np.random.randint(0, dim_b, size=a_var_nr)
            edge_to = np.random.randint(0, dim_b - 2, size=a_var_nr)
            edges = np.stack((edge_from, edge_to))
        edge_to = (edge_from + edge_to + 2) % dim_b
        a_var[edge_from, list(range(a_var_nr))] = 1
        a_var[edge_to, list(range(a_var_nr))] = -1
        a[i] = np.concatenate((a_base, a_var), axis=1)

        # change edge order of a so that (hopefully) the network can actually learn with all 11 dimensions (edges) of a
        np.random.shuffle(a[i].T)

        # calculate optimal solution
        sol[i] = solve_lp(c, a[i], b, method="highs-ds")

        valid_x = all_valid_subpaths(a[i], b)
        valid_x_sol = np.zeros((valid_x.shape[0], dim_x))
        for j in range(valid_x_sol.shape[0]):
            # calculate the optimal solution given certain edges
            valid_x_sol[j] = solve_lp_given_x(c, a[i], b, valid_x[j])

        # save
        disk_a = np.load(f"data/rs_{random_seed}_a.npy")
        disk_a = np.concatenate((disk_a, a[i:i + 1]), axis=0)
        np.save(f"data/rs_{random_seed}_a.npy", disk_a)

        disk_sol = np.load(f"data/rs_{random_seed}_sol.npy")
        disk_sol = np.concatenate((disk_sol, sol[i:i + 1]), axis=0)
        np.save(f"data/rs_{random_seed}_sol.npy", disk_sol)

        disk_x = np.load(f"data/rs_{random_seed}_x.npy")
        valid_x = np.concatenate((np.full((valid_x.shape[0], 1), i), valid_x), axis=1)  # to know for which a
        disk_x = np.concatenate((disk_x, valid_x), axis=0)
        np.save(f"data/rs_{random_seed}_x.npy", disk_x)

        disk_y = np.load(f"data/rs_{random_seed}_y.npy")
        valid_x_sol = np.concatenate((np.full((valid_x_sol.shape[0], 1), i), valid_x_sol), axis=1)  # to know which a
        disk_y = np.concatenate((disk_y, valid_x_sol), axis=0)
        np.save(f"data/rs_{random_seed}_y.npy", disk_y)

        sub_path_count += valid_x.shape[0]
        print(f"\rCalculated and saved {i + 1} data points ({sub_path_count} sub-paths).", end="")

    # remove the first (empty) rows of a, b, x, y
    disk_a = np.load(f"data/rs_{random_seed}_a.npy")[1:]
    np.save(f"data/rs_{random_seed}_a.npy", disk_a)

    disk_sol = np.load(f"data/rs_{random_seed}_sol.npy")[1:]
    np.save(f"data/rs_{random_seed}_sol.npy", disk_sol)

    disk_x = np.load(f"data/rs_{random_seed}_x.npy")[1:]
    np.save(f"data/rs_{random_seed}_x.npy", disk_x)

    disk_y = np.load(f"data/rs_{random_seed}_y.npy")[1:]
    np.save(f"data/rs_{random_seed}_y.npy", disk_y)

    disk_b = np.load(f"data/rs_{random_seed}_b.npy")
    disk_c = np.load(f"data/rs_{random_seed}_c.npy")

    # shapes which will be returned
    a = np.zeros((disk_x.shape[0], disk_a.shape[1], disk_a.shape[2]))
    sol = np.zeros((disk_x.shape[0], disk_sol.shape[1]))
    x = np.zeros((disk_x.shape[0], disk_x.shape[1] - 1))
    y = np.zeros((disk_y.shape[0], disk_y.shape[1] - 1))

    # the first column in x and y links to the corresponding graph (a), this is done to save storage for a
    # here, create a so that is has as many instances as x and that a, sol, x, and y can be indexed together
    for index in range(disk_x.shape[0]):
        a_index = int(disk_x[index, 0])
        a[index] = disk_a[a_index]
        sol[index] = disk_sol[a_index]
        x[index] = disk_x[index, 1:]
        y[index] = disk_y[index, 1:]

    # shuffle
    np.random.seed(random_seed)
    indices = np.random.permutation(x.shape[0])
    a = a[indices]
    x = x[indices]
    y = y[indices]
    sol = sol[indices]

    return disk_c, a, disk_b, x, y, sol


def solve_ks(c, a, b):
    """ Solve a knapsack problem using the ortools solver. """
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, "KnapsackSolver")

    values = c.tolist()
    weights = a.tolist()
    capacities = b.tolist()

    solver.Init(values, weights, capacities)
    solver.Solve()
    sol = [1 if solver.BestSolutionContains(i) else 0 for i in range(len(values))]
    return np.array(sol)


def generate_knapsack(dim_x, random_seed, num_x, vary_c, vary_a, vary_b, solve, max_val=1000):
    """ I will directly start with the MPLP and general LP setup, skipping the single LP setup, as a SingleLP either is
    so small, that learning can be easily done by overfitting, or a very large and complex problem would have to be
    generated which I do not want to do when first looking at a new problem.
    Note, that this method was written in a way that makes it so that the classes y are, on average, balanced. However,
    this is only true overall. So for specific random seeds, a set matrix A might result in very unbalanced matrices of
    generated data with respect to y (because of how x, a, and b function together).  TODO I think that is a problem, improve
    max_val sets the highest possible item weight and value to be generated (the knapsack can be larger). """
    # This function tries to save time by saving data (if newly generated) and loading data (if previously generated)
    param_string = f"{dim_x}_{random_seed}_{num_x}_{vary_c}_{vary_a}_{vary_b}_{solve}_{max_val}"
    save_folder = "ks_" + str(param_string)
    try:
        c = np.load(f"data/{save_folder}/c.npy")
        a = np.load(f"data/{save_folder}/a.npy")
        b = np.load(f"data/{save_folder}/b.npy")
        x = np.load(f"data/{save_folder}/x.npy")
        y = np.load(f"data/{save_folder}/y.npy")
        sol = np.load(f"data/{save_folder}/sol.npy")
        return c, a, b, x, y, sol
    except FileNotFoundError:
        pass

    np.random.seed(random_seed)
    dim_b = 1
    randint_high = max_val + 1
    randint_min = 1

    # c can be generated randomly without considering the other values
    if not vary_c:
        # one c for all generated instances
        c = np.random.randint(randint_min, randint_high, size=dim_x)
    else:
        # one c for each instance
        c = np.random.randint(randint_min, randint_high, size=(num_x, dim_x))

    sol = None

    if not vary_a:
        # one a for all generated instances
        a = np.random.randint(randint_min, randint_high, size=(dim_b, dim_x))

        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.randint(0, 2, (num_x, dim_x))

            # set b so that x has an about even chance of being within the space of valid solutions
            b = np.array([int(np.sum(a)/2)])

            # solving the LP(s)
            if solve:
                if not vary_c:
                    sol = solve_ks(c, a, b)
                else:
                    sol = np.zeros((num_x, dim_x), dtype=np.int32)
                    for i in range(num_x):
                        sol[i] = solve_ks(c[i], a, b)

        else:
            # one b for each instance

            # generate x
            x = np.random.randint(0, 2, (num_x, dim_x))

            # set b, since dim_b is always one, this gives about 50/50 for y
            b = np.random.randint(randint_min, randint_high, size=(num_x, dim_b)) * (dim_x / 2)

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x), dtype=np.int32)
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_ks(c, a, b[i])
                    else:
                        sol[i] = solve_ks(c[i], a, b[i])
    else:
        # one a for each instance
        if not vary_b:
            # one b for all generated instances

            # generate x
            x = np.random.randint(0, 2, (num_x, dim_x))

            # generate a
            a = np.random.randint(randint_min, randint_high, size=(num_x, dim_b, dim_x))

            # set b so that x has an about even chance of being within the space of valid solutions
            # this should on average give an even chance, but depending on seed results might be strongly unbalanced
            b = np.random.randint(randint_min, randint_high, size=dim_b) * (dim_x/2)

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x), dtype=np.int32)
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_ks(c, a[i], b)
                    else:
                        sol[i] = solve_ks(c[i], a[i], b)

        else:
            # one b for each instance

            # generate a
            a = np.random.randint(randint_min, randint_high, size=(num_x, dim_b, dim_x))

            # generate x
            x = np.random.randint(0, 2, (num_x, dim_x))

            # set b, since dim_b is always one, this gives about 50/50 for y
            b = np.random.randint(randint_min, randint_high, size=(num_x, dim_b)) * (dim_x/2)

            # solving the LPs
            if solve:
                sol = np.zeros((num_x, dim_x), dtype=np.int32)
                for i in range(num_x):
                    if not vary_c:
                        sol[i] = solve_ks(c, a[i], b[i])
                    else:
                        sol[i] = solve_ks(c[i], a[i], b[i])

    y = label_x_feasible(x, a, b)

    # save data to be able to load it next time
    os.makedirs(f"data/{save_folder}")
    np.save(f"data/{save_folder}/c.npy", c)
    np.save(f"data/{save_folder}/a.npy", a)
    np.save(f"data/{save_folder}/b.npy", b)
    np.save(f"data/{save_folder}/x.npy", x)
    np.save(f"data/{save_folder}/y.npy", y)
    np.save(f"data/{save_folder}/sol.npy", sol)

    return c, a, b, x, y, sol


# test functions from this file here
if __name__ == "__main__":
    # seed2 = 0
    # c2, a2, b2, x2, y2, sol2 = generate_knapsack(7, seed2, 10000, True, True, True, True)
    # print(np.sum(y2))

    # generate_simple_lp_v2(2, 3, 0, 100000, False, False, False, False)
    # generate_simple_lp_v3(3, 1, 0, 100000, False, False, False, False)
    # generate_simple_lp_v3(3, 30, 0, 10000, False, False, False, False)

    # toy_sp1b(100, 0)
    pass


# TODO general idea: it would seem better to always generate b randomly and then generate x depending on that
# TODO --> what about setting x on the class boundary, outwards going gaussian distributed? I think that is a good idea,
# but is is not trivial how to do that (at least not for me)
