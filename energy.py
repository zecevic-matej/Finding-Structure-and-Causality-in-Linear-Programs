import argparse

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

from captum.attr import IntegratedGradients


def temp_make_energy():
    """ I tried to formulate the energy problem as a LP formulation, so defining c, A, and b. I am almost entirely
    certain that this is in fact possible but I had too many problems and did not want to invest to much time in this.
    So I stopped trying. """
    time = 8760

    # inputs:
    c_pv = 1
    c_bat = 1
    c_buy = 1
    demand = np.genfromtxt("data/TS_Demand.csv").reshape((time, 1))
    avail_pv = np.genfromtxt("data/TS_PVAvail.csv").reshape((time, 1))

    # cost vector
    c_buy = np.full(time, c_buy)
    c_zeros = np.zeros(time*5)  # for the right dimensionality
    c = np.concatenate((np.array([c_pv]), np.array([c_bat]), c_buy, c_zeros))
    x_dim = c.shape[0]

    # how does x look like? (everything with "(T)" contains T (=time) elements, one for each time step)
    # c_ap_pv, c_ap_bat_s, p_buy (T), p_pv (T), p_bat_out (T), p_bat_in (T), p_bat_s (T)

    # constraints

    # energy balance (maybe remove the equality and make it an inequality (>= Demand, or <= - Demand only)
    a_energy_balance = sparse.lil_matrix((2*time, x_dim))
    b_energy_balance = sparse.lil_matrix((2*time, 1))
    for t in range(time):  # this can definitely be written more efficiently, for now I just want it correct though
        a_energy_balance[t * 2, 2 + t] = 1  # p_buy(t)
        a_energy_balance[t * 2, 2 + time + t] = 1  # p_pv(t)
        a_energy_balance[t * 2, 2 + 2 * time + t] = 1  # p_bat_out(t)
        a_energy_balance[t * 2, 2 + 3 * time + t] = -1  # p_bat_in(t)
        b_energy_balance[t * 2] = demand[t]

        a_energy_balance[t * 2 + 1, 2 + t] = -1  # p_buy(t)
        a_energy_balance[t * 2 + 1, 2 + time + t] = -1  # p_pv(t)
        a_energy_balance[t * 2 + 1, 2 + 2 * time + t] = -1  # p_bat_out(t)
        a_energy_balance[t * 2 + 1, 2 + 3 * time + t] = 1  # p_bat_in(t)
        b_energy_balance[t * 2 + 1] = -demand[t]

    # battery equation
    a_battery_equation = sparse.lil_matrix((2 * (time - 1), x_dim))
    b_battery_equation = sparse.lil_matrix((2 * (time - 1), 1))  # just stays zero, so that is fine
    for t in range(1, time):
        a_battery_equation[(t - 1) * 2, 2 + 4 * time + t] = 1  # p_bat_s (t)
        a_battery_equation[(t - 1) * 2, 2 + 4 * time + t - 1] = -1  # p_bat_s (t - 1)
        a_battery_equation[(t - 1) * 2, 2 + 3 * time + t] = -1  # p_bat_in (t)
        a_battery_equation[(t - 1) * 2, 2 + 2 * time + t] = 1  # p_bat_out (t)

        a_battery_equation[(t - 1) * 2 + 1, 2 + 4 * time + t] = -1  # p_bat_s (t)
        a_battery_equation[(t - 1) * 2 + 1, 2 + 4 * time + t - 1] = 1  # p_bat_s (t - 1)
        a_battery_equation[(t - 1) * 2 + 1, 2 + 3 * time + t] = 1  # p_bat_in (t)
        a_battery_equation[(t - 1) * 2 + 1, 2 + 2 * time + t] = -1  # p_bat_out (t)

    # pv production limit (0 <= p_pv (t) always given per LP definition (x >= 0))
    # lifetime missing (delta t I think) --> but why not for battery (per slides: not, per code: yes)
    a_pv_production_limit = sparse.lil_matrix((time, x_dim))
    b_pv_production_limit = sparse.lil_matrix((time, 1))  # just stays zero, so that is fine
    for t in range(time):
        a_pv_production_limit[t, 2 + time + t] = 1
        a_pv_production_limit[t, 0] = -avail_pv[t]

    # battery charge limit (0 <= p_bat_in (t) always given per LP definition (x >= 0))
    a_battery_charge_limit = sparse.lil_matrix((time, x_dim))
    b_battery_charge_limit = sparse.lil_matrix((time, 1))  # just stays zero, so that is fine
    for t in range(time):
        a_battery_charge_limit[t, 2 + 2 * time + t] = 1
        a_battery_charge_limit[t, 1] = -1

    # battery initial state
    a_battery_initial_state = sparse.lil_matrix((2, x_dim))
    b_battery_initial_state = sparse.lil_matrix((2, 1))  # just stays zero, so that is fine
    a_battery_initial_state[0, 2 + 4 * time] = 1
    a_battery_initial_state[0, 2 + 4 * time] = -1  # maybe not necessary because of x >= 0

    # power purchase limit (0 <= p_buy (t) always given per LP definition (x >= 0))

    # concatenate for constraint matrix a and vector b
    a = sparse.vstack(
        (a_energy_balance, a_battery_equation, a_pv_production_limit, a_battery_charge_limit, a_battery_initial_state))
    b = sparse.vstack(
        (b_energy_balance, b_battery_equation, b_pv_production_limit, b_battery_charge_limit, b_battery_initial_state))

    # time with sparse matrices: csr: 92.63453531265259, lil: 1.463003396987915

    # the following calculation of the solution (linprog): took like two hours, did not finish I stopped it, maybe I
    # could use more linprog options, or I just don't use this (better I think)
    res = linprog(c, a, b.toarray(), method="highs")
    sol = res.x
    print("Finished")


def load_energy_data(seed=0, with_new=True):
    """ Load data points for the energy system. Currently contains 10000 data points. """
    # data has 9 columns: cost_pv, cost_bat, cost_buy, demand, cap_pv, cap_bat, own_gen, totex, capex
    orig_data = np.load("data/energy_data.npy")

    orig_data_plus = None
    if with_new:
        # "data_plus.npy" is the combination of "data_values_interval.npy", data_values_around_min.npy", and
        # "data_values_around_max.npy", generated by "energy_data.py"
        orig_data_plus = np.load("data/data_plus.npy")
        orig_data = np.concatenate((orig_data, orig_data_plus))

    np.random.seed(seed)
    np.random.shuffle(orig_data)

    # this manually set mean is really (!) close to the calculated mean anyway (with infinite data, it would be exactly
    # the same) but using this manually set means means we have data for +1 and -1 (normalized) for all input values
    # when also using the newly generated data
    if with_new:
        # orig_data_plus[100] is exactly the mean data with corresponding outputs
        data_mean = orig_data_plus[100]  # np.mean(orig_data, axis=0)
    else:
        data_mean = np.mean(orig_data, axis=0)
    # normalize data and save variables to be able to reverse that
    data = orig_data - data_mean
    data_max = np.max(np.abs(data), axis=0)
    data = data / data_max

    e_input = data[:, :4]
    e_output = data[:, 4:]

    return e_input, e_output, data_mean, data_max, orig_data


def vis_energy(attributions, values=None, edge_labels=False, vis_real=False):
    """ Visualizes attribution for the energy neural network, from inputs to outputs. """
    inp_params, output_vals, pred_outputs = None, None, None
    if values is not None:
        inp_params, output_vals, pred_outputs = values
    g = nx.DiGraph()

    # index to name (input/output)
    itni = {0: "Photovoltaik", 1: "Batteriespeicher", 2: "Stromnetz", 3: "Energiebedarf"}
    itno = {0: "Kapazität PV", 1: "Kapazität Batterie", 2: "Eigenerzeugung", 3: "TOTEX", 4: "CAPEX"}

    # define nodes
    g.add_node("Photovoltaik", pos=(0, 7))
    g.add_node("Batteriespeicher", pos=(0, 5))
    g.add_node("Stromnetz", pos=(0, 3))
    g.add_node("Energiebedarf", pos=(0, 1))

    g.add_node("Kapazität PV", pos=(5, 8))
    g.add_node("Kapazität Batterie", pos=(5, 6))
    g.add_node("Eigenerzeugung", pos=(5, 4))
    g.add_node("TOTEX", pos=(5, 2))
    g.add_node("CAPEX", pos=(5, 0))

    labeldict = {}
    if values is not None:
        labeldict["Photovoltaik"] = f"Photovoltaik\n{inp_params[0, 0]:.2f}"
        labeldict["Batteriespeicher"] = f"Batteriespeicher\n{inp_params[0, 1]:.2f}"
        labeldict["Stromnetz"] = f"Stromnetz\n{inp_params[0, 2]:.2f}"
        labeldict["Energiebedarf"] = f"Energiebedarf\n{inp_params[0, 3]:.2f}"
        if output_vals is None:
            labeldict["Kapazität PV"] = f"Kapazität PV\n{pred_outputs[0, 0]:.2f}"
            labeldict["Kapazität Batterie"] = f"Kapazität Batterie\n{pred_outputs[0, 1]:.2f}"
            labeldict["Eigenerzeugung"] = f"Eigenerzeugung\n{pred_outputs[0, 2]:.2f}"
            labeldict["TOTEX"] = f"TOTEX\n{pred_outputs[0, 3]:.2f}"
            labeldict["CAPEX"] = f"CAPEX\n{pred_outputs[0, 4]:.2f}"
        else:
            labeldict["Kapazität PV"] = f"Kapazität PV\n{pred_outputs[0, 0]:.2f} ({output_vals[0, 0]:.2f})"
            labeldict["Kapazität Batterie"] = f"Kapazität Batterie\n{pred_outputs[0, 1]:.2f} ({output_vals[0, 1]:.2f})"
            labeldict["Eigenerzeugung"] = f"Eigenerzeugung\n{pred_outputs[0, 2]:.2f} ({output_vals[0, 2]:.2f})"
            labeldict["TOTEX"] = f"TOTEX\n{pred_outputs[0, 3]:.2f} ({output_vals[0, 3]:.2f})"
            labeldict["CAPEX"] = f"CAPEX\n{pred_outputs[0, 4]:.2f} ({output_vals[0, 4]:.2f})"
        if vis_real:
            _, _, data_mean, data_max, _ = load_energy_data()
            str_add = "\n" + r"$\rightarrow$"
            input_diffs = inp_params[0, :] * data_max[:4]
            real_inputs = input_diffs + data_mean[:4]
            labeldict["Photovoltaik"] += str_add + f"{real_inputs[0]:.0f} ({'+' if input_diffs[0] > 0 else ''}{input_diffs[0]:.0f})"
            labeldict["Batteriespeicher"] += str_add + f"{real_inputs[1]:.0f} ({'+' if input_diffs[1] > 0 else ''}{input_diffs[1]:.0f})"
            labeldict["Stromnetz"] += str_add + f"{real_inputs[2]:.0f} ({'+' if input_diffs[2] > 0 else ''}{input_diffs[2]:.3f})"
            labeldict["Energiebedarf"] += str_add + f"{real_inputs[3]:.0f} ({'+' if input_diffs[3] > 0 else ''}{input_diffs[3]:.0f})"
            output_diffs = pred_outputs[0, :] * data_max[4:]
            real_outputs = output_diffs + data_mean[4:]
            labeldict["Kapazität PV"] += str_add + f"{real_outputs[0]:.2f} ({'+' if output_diffs[0] > 0 else ''}{output_diffs[0]:.2f})"
            labeldict["Kapazität Batterie"] += str_add + f"{real_outputs[1]:.2f} ({'+' if output_diffs[1] > 0 else ''}{output_diffs[1]:.2f})"
            labeldict["Eigenerzeugung"] += str_add + f"{real_outputs[2]:.3f} ({'+' if output_diffs[2] > 0 else ''}{output_diffs[2]:.3f})"
            labeldict["TOTEX"] += str_add + f"{real_outputs[3]:.0f} ({'+' if output_diffs[3] > 0 else ''}{output_diffs[3]:.0f})"
            labeldict["CAPEX"] += str_add + f"{real_outputs[4]:.0f} ({'+' if output_diffs[4] > 0 else ''}{output_diffs[4]:.0f})"
    else:
        labeldict["Photovoltaik"] = "Photovoltaik"
        labeldict["Batteriespeicher"] = "Batteriespeicher"
        labeldict["Stromnetz"] = "Stromnetz"
        labeldict["Energiebedarf"] = "Energiebedarf"
        labeldict["Kapazität PV"] = "Kapazität PV"
        labeldict["Kapazität Batterie"] = "Kapazität Batterie"
        labeldict["Eigenerzeugung"] = "Eigenerzeugung"
        labeldict["TOTEX"] = "TOTEX"
        labeldict["CAPEX"] = "CAPEX"

    edge_list = []
    edge_attr = []
    for i, o in attributions:
        edge_list.append((itni[i], itno[o]))
        edge_attr.append(attributions[i, o])

    color_bounds = np.max(np.abs(edge_attr))
    cmap = plt.cm.RdBu

    pos = nx.get_node_attributes(g, "pos")

    nx.draw(g, pos, labels=labeldict, with_labels=True, node_color="w")

    nx.draw_networkx_edges(g, pos, edgelist=edge_list, edge_color=edge_attr, edge_cmap=cmap, edge_vmin=-color_bounds,
                           edge_vmax=color_bounds)

    if edge_labels:
        e_labels = {(itni[key[0]], itno[key[1]]): f"{attributions[key]:.4f}" for key in attributions.keys()}
        nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=e_labels, label_pos=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-color_bounds, vmax=color_bounds))
    sm.set_array([])
    plt.colorbar(sm)

    plt.show()


def visualize_loss(train_loss, test_loss):
    """ Plot the train and test loss of a neural network after learning. One graph shows the loss progress over all
    iterations, another one for only the last 10 iterations (can see whether it is still improving)."""
    nr_epochs = list(range(len(train_loss)+1))[1:]
    print(train_loss)
    print(test_loss)

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(nr_epochs, train_loss, label="Train")
    ax1.plot(nr_epochs, test_loss, label="Test")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(nr_epochs)
    ax1.set_xticklabels(str(epoch) for epoch in nr_epochs)
    ax1.set_title("Loss over all epochs")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(nr_epochs[-10:], train_loss[-10:], label="Train")
    ax2.plot(nr_epochs[-10:], test_loss[-10:], label="Test")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(nr_epochs[-10:])
    ax2.set_xticklabels(str(epoch) for epoch in nr_epochs[-10:])
    ax2.set_title("Loss over all epochs")
    ax2.set_title("Loss over the last 10 epochs")
    ax2.grid(True)
    ax2.legend()

    plt.subplots_adjust(hspace=0.6)

    plt.show()


class EnergyNet(nn.Module):
    """ Neural network learning the relationship of the input and output values as loaded by load_energy_data. """

    def __init__(self, dim_input, dim_output):
        """ Initialize the neural network. """
        super(EnergyNet, self).__init__()
        factor = 80 * 2 * 2 * 2 * 2
        self.fc1 = nn.Linear(dim_input, factor * 2)
        self.fc2 = nn.Linear(factor * 2, factor * 2)
        self.fc2b = nn.Linear(factor * 2, factor * 2)
        self.fc3 = nn.Linear(factor * 2, factor)
        self.fc4 = nn.Linear(factor, dim_output)

    def forward(self, param):
        """ Forward pass. """
        h = self.fc1(param)
        h = func.relu(h)
        h = self.fc2(h)
        h = func.relu(h)
        h = self.fc2b(h)
        h = func.relu(h)
        h = self.fc3(h)
        h = func.relu(h)
        output = self.fc4(h)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """ Train the model. """
    model.train()
    # mean squared error loss for output
    criterion = torch.nn.MSELoss()
    for batch_idx, (e_input, e_output) in enumerate(train_loader):
        e_input, e_output = e_input.to(device), e_output.to(device)
        optimizer.zero_grad()
        prediction = model(e_input)
        loss = criterion(prediction, e_output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(e_input),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    """ Test the model. """
    model.eval()
    test_loss = 0
    # mean squared error loss for output
    criterion = torch.nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for (e_input, e_output) in test_loader:
            e_input, e_output = e_input.to(device), e_output.to(device)
            prediction = model(e_input)
            test_loss += criterion(prediction, e_output).item()

    test_loss /= len(test_loader.batch_sampler)

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    return test_loss


def train_model(args):
    """ Get model parameters, data and train a model. """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    e_input, e_output, _, _, _ = load_energy_data(args.seed)

    # parameter
    input_train = e_input[:e_input.shape[0] // 2]
    input_test = e_input[e_input.shape[0] // 2:]

    # use half of the data for training and testing each
    output_train = e_output[:e_output.shape[0] // 2]
    output_test = e_output[e_output.shape[0] // 2:]

    tensor_input_train = torch.Tensor(input_train)
    tensor_input_test = torch.Tensor(input_test)

    tensor_output_train = torch.Tensor(output_train)
    tensor_output_test = torch.Tensor(output_test)

    dataset_train = TensorDataset(tensor_input_train, tensor_output_train)
    dataset_test = TensorDataset(tensor_input_test, tensor_output_test)

    dataloader_train = DataLoader(dataset_train, **train_kwargs)
    dataloader_train_for_test = DataLoader(dataset_train, **test_kwargs)
    dataloader_test = DataLoader(dataset_test, **test_kwargs)

    model = EnergyNet(e_input.shape[1], e_output.shape[1]).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train_loss = []
    test_loss = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, dataloader_train, optimizer, epoch)
        train_loss.append(test(model, device, dataloader_train_for_test))
        test_loss.append(test(model, device, dataloader_test))
        scheduler.step()
    visualize_loss(train_loss, test_loss)
    print(train_loss)
    print(test_loss)

    if args.save_model:
        # save the model
        save_path = f"models/Energy_{args.save_name}.pt"
        torch.save(model.state_dict(), save_path)

    return model


def prepare_model(args):
    """ Define the model and load the state in the specified path. """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    e_input, e_output, _, _, _ = load_energy_data(args.seed)
    model = EnergyNet(e_input.shape[1], e_output.shape[1]).to(device)

    # load the model state
    save_path = f"models/Energy_{args.save_name}.pt"
    model.load_state_dict(torch.load(save_path))
    return model


def apply_visualization(model, args):
    """ Set all necessary parameters and call the right visualization method. """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    e_input, e_output, _, _, _ = load_energy_data(args.seed)
    input_test = e_input[e_input.shape[0] // 2:]

    output_test = e_output[e_output.shape[0] // 2:]

    start_index = args.num_vis * args.vis_next
    if start_index + args.num_vis > input_test.shape[0]:
        raise ValueError(f"There are not enough test instances to visualize with respect to \"args.num_vis\": "
                         f"{args.num_vis} and \"args.vis_next\": {args.vis_next}")

    input_test = input_test[start_index:start_index + args.num_vis]
    tensor_input_vis = torch.Tensor(input_test).to(device)
    output_test = output_test[start_index:start_index + args.num_vis]

    attributions = {}
    for i in range(args.num_vis):

        # custom input
        if args.vis_input:
            vis_input_str = input("Enter input values ([-1, 1], 4 values, whitespace separated):")
            for index, value in enumerate(vis_input_str.split()):
                tensor_input_vis[i, index] = float(value)

        for output_index in range(5):
            ig = IntegratedGradients(model)
            # this following code or the block after? What is better, and for what purpose? Both seem very similar
            # for input_index in range(4):
            #     if args.vis_only_input != -1 and args.vis_only_input != input_index:
            #         continue
            #     # baseline
            #     bl = tensor_input_vis[i:i + 1].detach().clone()
            #     bl[0, input_index] = 0
            #     attr = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
            #     attr = attr.detach().cpu().numpy()[0]
            #     if (input_index, output_index) in attributions:
            #         attributions[(input_index, output_index)] += attr[input_index]
            #     else:
            #         attributions[(input_index, output_index)] = attr[input_index]

            # which  baseline to use
            choose_baseline = args.baseline
            # baseline for all smallest and all largest inputs
            if choose_baseline == "edges":
                bl = tensor_input_vis[i:i + 1].detach().clone()
                bl[0, :] = -1
                attr = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
                attr = attr.detach().cpu().numpy()[0]
                bl[0, :] = 1
                attr2 = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
                attr += attr2.detach().cpu().numpy()[0]
            # random: multiple baselines, uniformly distributed within [-1, 1], average for final attribution
            elif choose_baseline == "random":
                all_attr = None
                for bls in range(10):
                    bl = ((torch.rand(1, 4) * 2) - 1).to(device)
                    attr = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
                    attr = attr.detach().cpu().numpy()[0]
                    if bls == 0:
                        all_attr = attr
                    else:
                        all_attr += attr
                attr = all_attr / 10
            # gaussian: multiple baselines, gaussian distributed around 0, average for final attribution
            elif choose_baseline == "gaussian":
                all_attr = None
                for bls in range(10):
                    std = 0.25  # pretty close to the underlying data std
                    bl = torch.normal(torch.tensor([[0.0, 0.0, 0.0, 0.0]]), std).to(device)
                    attr = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
                    attr = attr.detach().cpu().numpy()[0]
                    if bls == 0:
                        all_attr = attr
                    else:
                        all_attr += attr
                attr = all_attr / 10
            # baseline as specified in args
            else:
                bl = tensor_input_vis[i:i + 1].detach().clone()
                bl[0, :] = float(choose_baseline)
                attr = ig.attribute(tensor_input_vis[i:i + 1], baselines=bl, target=output_index)
                attr = attr.detach().cpu().numpy()[0]
            for input_index in range(4):
                if args.vis_only_input == -1 or args.vis_only_input == input_index:
                    attributions[(input_index, output_index)] = attr[input_index]
        if not args.vis_agg:
            pred = model(tensor_input_vis[i:i + 1]).detach().cpu().numpy()
            if args.vis_input:
                out_label = None
            else:
                out_label = output_test[i:i + 1]
            vis_energy(attributions, values=(tensor_input_vis[i:i + 1].detach().cpu().numpy(), out_label, pred),
                       edge_labels=args.vis_only_input != -1, vis_real=args.vis_real_values)
            attributions = {}

    if args.vis_agg:
        # pred = model(tensor_input_vis).detach().cpu().numpy()
        vis_energy(attributions, edge_labels=args.vis_only_input != -1, vis_real=args.vis_real_values)

    return


def prepare_arguments():
    """ Define and return arguments. """
    parser = argparse.ArgumentParser(description="PyTorch Energy Experiment v0_1")
    # model training
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    # model saving / loading
    parser.add_argument("--save-model", action="store_true", default=False, help="save the current model")
    parser.add_argument("--load-model", action="store_true", default=False, help="load a model")
    parser.add_argument("--save-name", type=str, default="0", metavar="NAME",
                        help="name with which the model will be saved or loaded")
    # visualization
    parser.add_argument("--vis", action="store_true", default=False, help="visualize model performance and attribution")
    parser.add_argument("--num-vis", type=int, default=10, metavar="N", help="number of instanced to be visualized")
    parser.add_argument("--vis-agg", action="store_true", default=False,
                        help="aggregate the attribution of all \"num-vis\" instances before the visualization)")
    parser.add_argument("--vis-next", type=int, default=0, metavar="N",
                        help="skips the first vis_next * num_vis instances, can visualize other instances that way")
    parser.add_argument("--vis-save", action="store_true", default=False,
                        help="save the visualization, otherwise simply show it")
    parser.add_argument("--vis-input", action="store_true", default=False,
                        help="enter own inputs for the visualization")
    parser.add_argument("--baseline", type=str, default="0", metavar="NAME OR NUMBER",
                        help="which baseline to use (\"edges\", \"random\", or a number as the baseline)")
    parser.add_argument("--vis-real-values", action="store_true", default=False,
                        help="also show the unnormalized values on the visualization")
    parser.add_argument("--vis-only-input", type=int, default=-1, metavar="N", help="only visualize for specific input")

    args = parser.parse_args()

    return args


def main():
    """ Run the neural network with the specified arguments. """
    # get arguments
    args = prepare_arguments()

    # get the model
    if not args.load_model:
        # train the model
        model = train_model(args)
    else:
        # load the model
        model = prepare_model(args)

    # obtain and visualize attributions
    if args.vis:
        apply_visualization(model, args)


if __name__ == "__main__":
    main()
