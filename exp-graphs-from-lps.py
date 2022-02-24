import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import os
import networkx as nx

import linear_programs as lp


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# START OF MY STUFF

# Used this in MPLP.py

# def apply_notears(model, info):  # quickly implemented, "bad" code (just so that it works, not parametric or anything)
#     import numpy as np
#     from notears import notears_linear
#
#     # # shortest path 2:
#     # # Following code: first 5 edges increase cost sixth not (good so far), why some later ones as well?
#     # # cost = np.matmul(info.sol, info.c) * -1
#     # # notears_input = np.concatenate((info.sol, cost.reshape((len(cost), 1))), 1)
#     #
#     # # set 1: is there an edge from A to B (0), A to C (1), ..., B to A (5), B to C (6), ..., F to E (29)
#     # a_other_form = np.zeros((info.a.shape[0], 30))
#     # for i in range(info.a.shape[0]):
#     #     for j in range(info.a.shape[2]):
#     #         start = np.where(info.a[i, :, j] == 1)[0]
#     #         end = np.where(info.a[i, :, j] == -1)[0]
#     #         if end > start:
#     #             end -= 1
#     #         index = start*5 + end
#     #         a_other_form[i, index] = 1
#     # notears_input = np.concatenate((a_other_form[:, 4:6], info.sol[:, :6]), 1)
#     # w_est = notears_linear(notears_input, lambda1=0.1, loss_type='l2', w_threshold=0)  # X: number_of_samples times feature_size
#     # print("NOTEARS")
#
#     # # MPLP param b
#     # notears_input = np.concatenate((info.x, info.y), 1)  # --> would we not expect this to give some causal relations? why doesn't it?
#     # notears_input = np.concatenate((info.b, info.y), 1)  # --> thats kind of interesting, shows my data generation a little (or does it? maybe rather has something to do with A)
#     # notears_input = np.concatenate((info.b, info.sol), 1)  # --> let's look at this again and see if I can find why this could make sense by looking at A and c (I did, it makes sense, let's get back to it later) TODO start here
#     # w_est = notears_linear(notears_input, lambda1=0.1, loss_type='l2', w_threshold=0)  # X: number_of_samples times feature_size
#     # print("NOTEARS")

# along with this in main()
#     # obtain and visualize attributions
#     if args.vis:
#         apply_visualization(model, info, args)
#     else:  # TODO this is just to look at it, the code style should be improved before adding and committing like this
#         apply_notears(model, info)


import matplotlib.pyplot as plt
plt.set_cmap('RdBu')
#import seaborn as sns
import gc


def clean_plt():
    plt.close('all')
    gc.collect()


def plot_all_individual(l_mat, l_title, suptitle, alt_form=None, alt_size=None, ax_labels=None, lines=None, other=None,
                        special_case=0, save_fig=None):
    clean_plt()
    if alt_form:
        y, x = alt_form
    else:
        y = 2
        x = 4
    if alt_size:
        size_x, size_y = alt_size
    else:
        size_x, size_y = (13, 8)
    fig, axs = plt.subplots(y, x, figsize=(size_x, size_y))

    for ind, a in enumerate(axs.flatten()):
        im = a.imshow(l_mat[ind], vmin=-1, vmax=1)
        a.set_title(l_title[ind], fontsize=13)
        if other is not None and ind in other:
            for i in range(l_mat[ind].shape[0]):
                for j in range(l_mat[ind].shape[1]):
                    # set the text of the matrix fields
                    im.axes.text(j, i, "{:.2f}".format(l_mat[ind][i, j]), va="center", ha="center")
            a.set_xticks([])
            a.set_yticks([])
            continue
        if ax_labels is not None:
            if special_case == 0:
                a.set_xticks(np.arange(l_mat[ind].shape[0]))
                a.set_yticks(np.arange(l_mat[ind].shape[1]))
                a.set_xticklabels(ax_labels)
                a.set_yticklabels(ax_labels)
            elif special_case == 1:
                a.set_xticks(np.array([2, 12, 21, 25]))
                a.set_yticks(np.arange(l_mat[ind].shape[1]))
                a.set_xticklabels(["c", "A", "b", "sol"], weight="bold", fontsize=12)
                a.set_yticklabels(ax_labels, fontsize=8)
            elif special_case == 2:
                ax_labels_less = [ax_l for count, ax_l in enumerate(ax_labels) if count % 6 == 0]

                a.set_xticks(np.arange(l_mat[ind].shape[0]), minor=True)
                a.set_xticks(np.arange(l_mat[ind].shape[1], step=6))
                a.set_xticklabels(ax_labels_less, fontsize=9)
                a.set_yticks(np.arange(l_mat[ind].shape[1]), minor=True)
                a.set_yticks(np.arange(l_mat[ind].shape[1], step=6))
                a.set_yticklabels(ax_labels_less, fontsize=9)

                # why does that not work? I solved it with the code above but still
                # a.set_xticks([])
                # a.set_xlabel("A", weight="bold", fontsize=12)
                # a.set_yticks(np.arange(l_mat[ind].shape[1], step=6))
                # ax_labels = [ax_l for count, ax_l in enumerate(ax_labels) if count % 6 == 0]
                # a.set_yticklabels(ax_labels)

            elif special_case == 3:
                ax_labels_less = [ax_l for count, ax_l in enumerate(ax_labels) if count % 6 == 0]
                ax_labels_less = ax_labels_less[:-2] + [r"$\bf{sol}$"]

                a.set_xticks(np.arange(l_mat[ind].shape[0]), minor=True)
                a.set_xticks(np.concatenate((np.arange(l_mat[ind].shape[1], step=6)[:-2], np.array([77.5]))))
                a.set_xticklabels(ax_labels_less, fontsize=9)
                a.set_yticks(np.arange(l_mat[ind].shape[1]), minor=True)
                a.set_yticks(np.concatenate((np.arange(l_mat[ind].shape[1], step=6)[:-2], np.array([77.5]))))
                a.set_yticklabels(ax_labels_less, fontsize=9)

            elif special_case == 4:
                a.set_xticks(np.arange(l_mat[ind].shape[0]), minor=True)
                a.set_xticks(np.array([5.5, 17.5]))
                a.set_xticklabels(["c", "sol"], weight="bold", fontsize=12)
                a.set_yticks(np.arange(l_mat[ind].shape[1]), minor=True)
                a.set_yticks(np.array([5.5, 17.5]))
                a.set_yticklabels(["c", "sol"], weight="bold", fontsize=12)

            elif special_case == 5:
                a.set_xticks(np.arange(l_mat[ind].shape[0]), minor=True)
                a.set_xticks(np.array([4.5, 14.5]))
                a.set_xticklabels(["x", "y"], weight="bold", fontsize=12)
                a.set_yticks(np.arange(l_mat[ind].shape[1]), minor=True)
                a.set_yticks(np.array([4.5, 14.5]))
                a.set_yticklabels(["x", "y"], weight="bold", fontsize=12)

        if lines is not None:
            for line in lines:
                a.plot([-0.5, -0.5 + l_mat[ind].shape[0]], [-0.5 + line, -0.5 + line], c="k", linewidth=3)
                a.plot([-0.5 + line, -0.5 + line], [-0.5, -0.5 + l_mat[ind].shape[1]], c="k", linewidth=3)

    plt.set_cmap('RdBu')
    plt.suptitle(suptitle, fontsize=18)
    if save_fig is not None:
        plt.savefig(save_fig)
    else:
        plt.show()
    clean_plt()


def draw_graph(matrix, nodes):
    color_bounds = np.max(np.abs(matrix))
    cmap = plt.cm.RdBu

    g = nx.DiGraph()

    for node in nodes:
        g.add_node(node)

    pos = nx.spring_layout(g)

    nx.draw(g, pos, with_labels=True)

    edge_list = []
    edge_color = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                edge_list.append((nodes[i], nodes[j]))
                edge_color.append(matrix[i, j])

    # connectionstyle="arc3, rad = 0.1",
    nx.draw_networkx_edges(g, pos, edgelist=edge_list, edge_color=edge_color, edge_cmap=cmap, edge_vmin=-color_bounds,
                           edge_vmax=color_bounds)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-color_bounds, vmax=color_bounds))
    sm.set_array([])
    plt.colorbar(sm)

    plt.show()

    plt.clf()
    plt.close()


if __name__ == '__main__':
    # show = [-1]
    #show = list(range(0, 100))  # go over all (there are less than 100)
    show = [5]
    base = "data/exp-graphs-from-lps/"
    if -1 in show:
        import notears.utils as utils
        utils.set_random_seed(1)

        n, d, s0, graph_type, sem_type = 100, 9, 9, 'ER', 'gauss'
        B_true = utils.simulate_dag(d, s0, graph_type)
        plot_all_individual([B_true, B_true], ["T1", "T2"], "Test", (1, 2),
                            (8, 9), ax_labels=["1", "1", "1", "1", "1", "1", "1", "1", "1"], lines=[2, 7])

        draw_graph(B_true, ["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    if 0 in show:
        import notears.utils as utils
        utils.set_random_seed(1)

        n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
        B_true = utils.simulate_dag(d, s0, graph_type)

        try:
            W_true = np.genfromtxt("data/exp-graphs-from-lps/W_true.csv", delimiter=",")
        except OSError:
            W_true = utils.simulate_parameter(B_true)
            np.savetxt('data/exp-graphs-from-lps/W_true.csv', W_true, delimiter=',')

        try:
            X = np.genfromtxt("data/exp-graphs-from-lps/X.csv", delimiter=",")
        except OSError:
            X = utils.simulate_linear_sem(W_true, n, sem_type)
            np.savetxt('data/exp-graphs-from-lps/X.csv', X, delimiter=',')

        try:
            W_est = np.genfromtxt("data/exp-graphs-from-lps/W_est.csv", delimiter=",")
        except OSError:
            W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
            assert utils.is_dag(W_est)
            np.savetxt('data/exp-graphs-from-lps/W_est.csv', W_est, delimiter=',')

        acc = utils.count_accuracy(B_true, W_est != 0)

        try:
            W_est_no_thresh = np.genfromtxt("data/exp-graphs-from-lps/W_est_no_thresh.csv", delimiter=",")
        except OSError:
            W_est_no_thresh = notears_linear(X, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt('data/exp-graphs-from-lps/W_est_no_thresh.csv', W_est_no_thresh, delimiter=',')

        plot_all_individual([B_true, W_true, W_est_no_thresh, W_est], ["B-True", "W-True", "W-est (lambda=threshold=0)",
                                                                       "W-est (with lambda & threshold)"], "", (2, 2),
                            (8, 9))
        print(acc)

    if 1 in show:
        energy_data = np.load("energy/data/energy_data.npy")
        data = energy_data

        try:
            w_energy = np.genfromtxt("data/exp-graphs-from-lps/W_energy.csv", delimiter=",")
        except OSError:
            w_energy = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt('data/exp-graphs-from-lps/W_energy.csv', w_energy, delimiter=',')

        try:
            w_energy_thresh = np.genfromtxt("data/exp-graphs-from-lps/W_energy_thresh.csv", delimiter=",")
        except OSError:
            w_energy_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
            np.savetxt('data/exp-graphs-from-lps/W_energy_thresh.csv', w_energy_thresh, delimiter=',')

        try:
            w_energy_lambda = np.genfromtxt("data/exp-graphs-from-lps/W_energy_lambda.csv", delimiter=",")
        except OSError:
            w_energy_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
            np.savetxt('data/exp-graphs-from-lps/W_energy_lambda.csv', w_energy_lambda, delimiter=',')

        try:
            w_energy_thresh_lambda = np.genfromtxt("data/exp-graphs-from-lps/W_energy_thresh_lambda.csv", delimiter=",")
        except OSError:
            w_energy_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
            np.savetxt('data/exp-graphs-from-lps/W_energy_thresh_lambda.csv', w_energy_thresh_lambda, delimiter=',')

        # cost_pv, cost_bat, cost_buy, demand, cap_pv, cap_bat, own_gen, totex, capex
        ax_labels = ["$c_{PV}$", "$c_{Bat}$", "$c_{Buy}$", "Demand", "$Cap_{PV}$", "$Cap_{Bat}$", "Own-Gen", "Totex",
                     "Capex"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_energy, w_energy_thresh, w_energy_lambda, w_energy_thresh_lambda], titles, "Energy",
                            (2, 2), (12, 9), ax_labels=ax_labels, lines=[4], save_fig=os.path.join(base,"Exp1_1.png"))
        print("Now: visualize the result so that the directions do not matter (just showing one diagonal)")
        w_energy2 = np.zeros_like(w_energy)
        w_energy_thresh2 = np.zeros_like(w_energy_thresh)
        w_energy_lambda2 = np.zeros_like(w_energy_lambda)
        w_energy_thresh_lambda2 = np.zeros_like(w_energy_thresh_lambda)
        for i in range(w_energy.shape[0]):
            for j in range(w_energy.shape[1]):
                if i == j:
                    w_energy2[i, j] = w_energy[i, j]
                    w_energy_thresh2[i, j] = w_energy_thresh[i, j]
                    w_energy_lambda2[i, j] = w_energy_lambda[i, j]
                    w_energy_thresh_lambda2[i, j] = w_energy_thresh_lambda[i, j]
                elif i < j:
                    w_energy2[i, j] = w_energy[i, j] + w_energy[j, i]
                    w_energy_thresh2[i, j] = w_energy_thresh[i, j] + w_energy_thresh[j, i]
                    w_energy_lambda2[i, j] = w_energy_lambda[i, j] + w_energy_lambda[j, i]
                    w_energy_thresh_lambda2[i, j] = w_energy_thresh_lambda[i, j] + w_energy_thresh_lambda[j, i]
                else:
                    continue
        plot_all_individual([w_energy2, w_energy_thresh2, w_energy_lambda2, w_energy_thresh_lambda2], titles, "Energy",
                            (2, 2), (12, 9), ax_labels=ax_labels, lines=[4], save_fig=os.path.join(base,"Exp1_2.png"))
        # --> I do not understand these results. Especially that the input -> output area (top right) is empty is
        # unexpected (I think the acyclic criterion plays a role here); still. top left should be empty
        #draw_graph(w_energy_thresh_lambda, ax_labels)

    if 2 in show:
        c, a, b, x, y, sol = lp.generate_simple_lp(5, 3, 0, 1000, False, False, False, False)
        # using this will flip from red to blue since the y encoding has flipped #y = np.ones((len(y),1)) - y
        data = np.concatenate((x, y), 1)

        # if you flip the encoding with this, you get positive instead of negative values #data[:,-1] = np.ones(len(data))-data[:,-1]

        w_2 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
        print("1")

        w_2_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
        print("2")

        w_2_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
        print("3")

        w_2_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
        print("4")

        ax_labels = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$", "$y$"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_2, w_2_thresh, w_2_lambda, w_2_thresh_lambda], titles,
                            "Specific LP, feasible/not-feasible", (2, 2), (8, 9), ax_labels=ax_labels, lines=[5])#, save_fig=os.path.join(base,"Exp2_1.png"))
        # --> I would have expected some negative edges from x to why but I can see how this is not as simple and that
        # too many other factors play a role here (with lambda=0 we actually can see that)

        # plt.imshow(np.triu(np.ones((4,4))*2)+np.tril(np.ones((4,4))*-2), vmin=-.5, vmax=.5)
        # plt.set_cmap("RdBu")
        # plt.colorbar()
        # plt.show()

    if 3 in show:
        # with seed = 1 some edges are more clear, also works with seed = 0 though
        c, a, b, x, y, sol = lp.generate_simple_lp(5, 3, 1, 1000, False, False, True, False)
        data = np.concatenate((b, y), 1)

        w_3 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

        w_3_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

        w_3_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

        w_3_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

        ax_labels = ["$b_0$", "$b_1$", "$b_2$", "$y$"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_3, w_3_thresh, w_3_lambda, w_3_thresh_lambda], titles,
                            "LP with parametric b, feasible/not-feasible", (2, 2), (8, 9), ax_labels=ax_labels,
                            lines=[3], save_fig=os.path.join(base,"Exp3_1.png"))
        # --> I generate b dependant of y, so it would make sense to see that; seeing y -> b instead is surprising but
        # this also makes sense (and actually fits my code where I randomly "decide" what y should be and that set b);
        # still, the other direction should also hold but --> acyclic!!!, so this direction seems to be stronger

    if 4 in show:
        c, a, b, x, y, sol = lp.generate_simple_lp(5, 3, 1, 1000, False, False, True, True)
        data = np.concatenate((b, sol), 1)

        w_4 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

        w_4_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

        w_4_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

        w_4_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

        ax_labels = ["$b_0$", "$b_1$", "$b_2$", "$sol_0$", "$sol_1$", "$sol_2$", "$sol_3$", "$sol_4$"]
        c = c.reshape((c.shape[0], 1))
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$", "LP matrix $A$", "cost (gain) vector $c$"]
        plot_all_individual([w_4, w_4_thresh, w_4_lambda, w_4_thresh_lambda, a, c], titles,
                            "LP with parametric b, impact on solution", (3, 2), (8, 13), ax_labels=ax_labels, lines=[3],
                            other=[4, 5], save_fig=os.path.join(base,"Exp4_1.png"))
        # --> this is nice: if b1 is larger, we can get more gain from sol_1 and less from sol_4; sol_4 clearly benefits
        # more from b0 and b2 than b1; the left blocks seem not as interesting; for the right down block, it makes sense
        # to have red edges -> if one sol is higher, others should be smaller

    if 5 in show:
        for seed in [0, 1]:
            c, a, b, x, y, sol = lp.generate_simple_lp(5, 3, seed, 1000, True, False, False, True)
            data = np.concatenate((c, sol), 1)

            w_5 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

            w_5_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

            w_5_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

            w_5_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

            ax_labels = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$sol_0$", "$sol_1$", "$sol_2$", "$sol_3$",
                         "$sol_4$"]
            b = b.reshape((b.shape[0], 1))
            titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                      "$\lambda=0.1$, $\omega=0.3$", "LP matrix $A$", "LP vector $b$"]
            plot_all_individual([w_5, w_5_thresh, w_5_lambda, w_5_thresh_lambda, a, b], titles,
                                "LP with parametric c, impact on solution", (3, 2), (8, 13), ax_labels=ax_labels,
                                lines=[5], other=[4, 5], save_fig=os.path.join(base,f"Exp5_1_{seed}.png"))
        # --> seed 0: constraints for sol_4 and sol_5 seem low, so it makes sense that high gain here increases the
        # solution of the respective value (I would expect the diagonal of the top right to be blue more often in
        # general) but constraints of 4 and 5 are so similar that we won't seem them together much; same kind of
        # "incompatibility" with sol_4 and sol_5
        # --> seed 1: I think with these a and b, sol_0 will almost always be 0; we can see that sol4-sol2 and sol3-sol1
        # will rarely share a solution, which makes sense when looking at a

    if 6 in show:
        c, a, b, x, y, sol = lp.generate_simple_lp(5, 3, 0, 1000, True, True, True, True)
        a = a.reshape((a.shape[0], a.shape[1]*a.shape[2]))
        data = np.concatenate((c, a, b, sol), 1)

        try:
            w_6 = np.genfromtxt("data/exp-graphs-from-lps/w_6.csv", delimiter=",")
        except OSError:
            w_6 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_6.csv", w_6, delimiter=",")

        try:
            w_6_thresh = np.genfromtxt("data/exp-graphs-from-lps/w_6_thresh.csv", delimiter=",")
        except OSError:
            w_6_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_6_thresh.csv", w_6_thresh, delimiter=",")

        try:
            w_6_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_6_lambda.csv", delimiter=",")
        except OSError:
            w_6_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_6_lambda.csv", w_6_lambda, delimiter=",")

        try:
            w_6_thresh_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_6_thresh_lambda.csv", delimiter=",")
        except OSError:
            w_6_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_6_thresh_lambda.csv", w_6_thresh_lambda, delimiter=",")

        ax_labels = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$A_0$", "$A_1$", "$A_2$", "$A_3$", "$A_4$", "$A_5$",
                     "$A_6$", "$A_7$", "$A_8$", "$A_9$", "$A_{10}$", "$A_{11}$", "$A_{12}$", "$A_{13}$", "$A_{14}$",
                     "$b_0$", "$b_1$", "$b_2$", "$sol_0$", "$sol_1$", "$sol_2$", "$sol_3$", "$sol_4$"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                      "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_6, w_6_thresh, w_6_lambda, w_6_thresh_lambda], titles, "General LP without fixed values",
                            (2, 2), (8, 9), ax_labels=ax_labels, lines=[5, 20, 23], special_case=1, save_fig=os.path.join(base,"Exp6_1.png"))
        # --> looking good: this does not give any new/useful insights but it fits assumptions

    if 7 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp1", 1000, 0)
        data = np.concatenate((x, y), 1)

        w_7 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

        w_7_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

        w_7_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

        w_7_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

        ax_labels = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$", "$x_6$", "$y$"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_7, w_7_thresh, w_7_lambda, w_7_thresh_lambda], titles, "Fixed Shortest Path", (2, 2),
                            (8, 9), ax_labels=ax_labels, lines=[7], save_fig=os.path.join(base,"Exp7_1.png"))
        # --> the "bluest" one on top is the edge from E to F which, in this graph, must be part of a solution

    if 8 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp2", 1000, 0)
        data = np.concatenate((x, y), 1)

        w_8 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

        w_8_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

        w_8_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

        w_8_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

        ax_labels = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$", "$x_6$", "$x_7$", "$x_8$", "$x_9$",
                     "$x_{10}$", "$x_{11}$", "$y$"]
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_8, w_8_thresh, w_8_lambda, w_8_thresh_lambda], titles,
                            "Shortest path with 6 fixed and 6 variable edges", (2, 2), (8, 9), ax_labels=ax_labels,
                            lines=[12], save_fig=os.path.join(base,"Exp8_1.png"))
        # --> the first 6 fixed edges: near A-->B and E-->F is good, F-->A is bad, on average others are bad

    if 9 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp1", 1000, 0)
        a = a.reshape((a.shape[0], a.shape[1]*a.shape[2]))
        #data = np.concatenate((x, y), 1)
        data = a  # can't see y anyway, and y looks very boring  # data = np.concatenate((a, y), 1)

        try:
            w_9 = np.genfromtxt("data/exp-graphs-from-lps/w_9.csv", delimiter=",")
        except OSError:
            w_9 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_9.csv", w_9, delimiter=",")

        try:
            w_9_thresh = np.genfromtxt("data/exp-graphs-from-lps/w_9_thresh.csv", delimiter=",")
        except OSError:
            w_9_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_9_thresh.csv", w_9_thresh, delimiter=",")

        try:
            w_9_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_9_lambda.csv", delimiter=",")
        except OSError:
            w_9_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_9_lambda.csv", w_9_lambda, delimiter=",")

        try:
            w_9_thresh_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_9_thresh_lambda.csv", delimiter=",")
        except OSError:
            w_9_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_9_thresh_lambda.csv", w_9_thresh_lambda, delimiter=",")

        ax_labels = []
        for i in range(a.shape[1]):
            ax_labels.append("$A_{" + str(i) + "}$")
        # ax_labels.append("$y$")
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_9, w_9_thresh, w_9_lambda, w_9_thresh_lambda], titles,
                            "Shortest path with 6 fixed and 6 variable edges", (2, 2), (8, 9), ax_labels=ax_labels,
                            lines=[], special_case=2, save_fig=os.path.join(base,"Exp9_1.png"))
        # --> guaranteed edges pattern: if a_x has one values, a_{x+12} (the next node) often has the opposite

    if 10 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp2", 1000, 0)
        a = a.reshape((a.shape[0], a.shape[1]*a.shape[2]))
        data = np.concatenate((a, sol), 1)

        try:
            w_10 = np.genfromtxt("data/exp-graphs-from-lps/w_10.csv", delimiter=",")
        except OSError:
            w_10 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_10.csv", w_10, delimiter=",")

        try:
            w_10_thresh = np.genfromtxt("data/exp-graphs-from-lps/w_10_thresh.csv", delimiter=",")
        except OSError:
            w_10_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_10_thresh.csv", w_10_thresh, delimiter=",")

        try:
            w_10_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_10_lambda.csv", delimiter=",")
        except OSError:
            w_10_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_10_lambda.csv", w_10_lambda, delimiter=",")

        try:
            w_10_thresh_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_10_thresh_lambda.csv", delimiter=",")
        except OSError:
            w_10_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_10_thresh_lambda.csv", w_10_thresh_lambda, delimiter=",")

        ax_labels = []
        for i in range(a.shape[1]):
            ax_labels.append("$A_{" + str(i) + "}$")
        for i in range(sol.shape[1]):
            ax_labels.append("$sol_{" + str(i) + "}$")
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_10, w_10_thresh, w_10_lambda, w_10_thresh_lambda], titles,
                            "Shortest path with 6 fixed and 6 variable edges", (2, 2), (8, 9), ax_labels=ax_labels,
                            lines=[a.shape[1]], special_case=3, save_fig=os.path.join(base,"Exp10_1.png"))
        # --> the first 5 (fixed) edges tend to be active together, the other edges not sol_5 (6th) is irrelevant;
        # bottom left: 6 variable e. starting at A (A-matrix: 1), bottom right: 6 variable e. ending in F (A-matrix:-1);
        # also: small trend, that fixed edges are not taken if var edges exist (see values above diagonals)

    if 11 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp4", 1000, 0)
        data = np.concatenate((c, sol), 1)

        w_11 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)

        w_11_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)

        w_11_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)

        w_11_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)

        ax_labels = []
        for i in range(c.shape[1]):
            ax_labels.append("$c_{" + str(i) + "}$")
        for i in range(sol.shape[1]):
            ax_labels.append("$sol_{" + str(i) + "}$")
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_11, w_11_thresh, w_11_lambda, w_11_thresh_lambda], titles,
                            "Shortest path with varying costs and edges", (2, 2), (8, 9), ax_labels=ax_labels,
                            lines=[12], special_case=4, save_fig=os.path.join(base,"Exp11_1.png"))
        # (also tried out a, very big visualization, nothing new interesting (which wasn't seen in other examples))
        # --> pretty simple: lower cost, more likely to be part of the solution

    if 12 in show:
        c, a, b, x, y, sol = lp.get_toy_problem("sp5", 100, 0, ignore_warnings=True)
        data = np.concatenate((x, y), 1)[:1000]  # data = np.concatenate((x, y, sol), 1)[:1000]

        try:
            w_12 = np.genfromtxt("data/exp-graphs-from-lps/w_12.csv", delimiter=",")
        except OSError:
            w_12 = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_12.csv", w_12, delimiter=",")

        try:
            w_12_thresh = np.genfromtxt("data/exp-graphs-from-lps/w_12_thresh.csv", delimiter=",")
        except OSError:
            w_12_thresh = notears_linear(data, lambda1=0, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_12_thresh.csv", w_12_thresh, delimiter=",")

        try:
            w_12_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_12_lambda.csv", delimiter=",")
        except OSError:
            w_12_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0)
            np.savetxt("data/exp-graphs-from-lps/w_12_lambda.csv", w_12_lambda, delimiter=",")

        try:
            w_12_thresh_lambda = np.genfromtxt("data/exp-graphs-from-lps/w_12_thresh_lambda.csv", delimiter=",")
        except OSError:
            w_12_thresh_lambda = notears_linear(data, lambda1=0.1, loss_type='l2', w_threshold=0.3)
            np.savetxt("data/exp-graphs-from-lps/w_12_thresh_lambda.csv", w_12_thresh_lambda, delimiter=",")

        ax_labels = []
        for i in range(x.shape[1]):
            ax_labels.append("$x_{" + str(i) + "}$")
        for i in range(y.shape[1]):
            ax_labels.append("$y_{" + str(i) + "}$")
        # for i in range(sol.shape[1]):
        #     ax_labels.append("$sol_{" + str(i) + "}$")
        titles = ["$\lambda=0$, $\omega=0$", "$\lambda=0$, $\omega=0.3$", "$\lambda=0.1$, $\omega=0$",
                  "$\lambda=0.1$, $\omega=0.3$"]
        plot_all_individual([w_12, w_12_thresh, w_12_lambda, w_12_thresh_lambda], titles,
                            "Shortest path: Complete the path", (2, 2), (8, 9), ax_labels=ax_labels, lines=[11, 22],
                            special_case=5, save_fig=os.path.join(base,"Exp12_1.png"))
        # --> x-y makes sense; also: here it actually matters that x->y and not y->x

# TODO (not TODO) there are some small things, I could look at (like x and y for sp4) but I think the existing examples
# TODO (not TODO) cover what's interesting the others ones would not be as interesting; keep that in mind though
# TODO (not TODO) One maybe: sp3 (very big, takes long and more imporantly: not great for visualization)