import json
import pickle
import sys
import time

import numpy as np
import torch
import pylab as plt
from scipy.io import loadmat
from tqdm import tqdm

from AngularPropTorch.angular_propagate_pytorch import PropagatePadded
from tools import OpticalLayer


def load_all_target_intensities(sim_args):
    """
    Transforms saved kernels to be appropriately scaled and padded for the simulation grid
    Since both negative and positive kernels are given separately, this function returns them separately.

    Returns:
        Two python lists for positive and negative target intensity profiles with the simulation grid. The energy in
        each intensity profile is normalized to 1.
    """

    kernel_ps, kernel_ns = load_kernels()

    def transform_kernel(kernel):
        """Changes the shape and number of pixels for the kernel to match the propagation grid by scaling and padding"""
        # turns a 1x1 pixel into a grid_elements_per_output x grid_elements_per_output sized box
        kernel = np.repeat(kernel, sim_args['grid_elements_per_output'], axis=0)
        kernel = np.repeat(kernel, sim_args['grid_elements_per_output'], axis=1)

        # pads the remaining spacing around the target kernel
        starting_size = np.shape(kernel)[0]
        pad = (sim_args['grid_n'] - starting_size) // 2
        kernel = np.pad(kernel, pad)
        return kernel

    # separated lists of target intensities for both positive and negative kernels
    kernels_p = []
    kernels_n = []

    n_kernels = np.shape(kernel_ps)[2]  # number of kernels

    def normalize(t):
        """Normalizes a phasor array so the cumulative intensity in the field is 1."""
        mag = np.sqrt(np.sum(np.abs(t) ** 2))
        t /= mag
        return t

    # transform both the positive and negative kernels to fit simulation grid
    for i in range(n_kernels):  # for each kernel in the file
        kernels_p.append(normalize(transform_kernel(kernel_ps[:, :, i, 0])))
        kernels_n.append(normalize(transform_kernel(kernel_ns[:, :, i, 0])))

    return kernels_p, kernels_n


def loss_fn(output, target):
    """
    :param output: The output activations from the model
    :param target: The target activations for the model
    :return: a scalar the is proportional to how "different" the input is from the output
    """

    def norm(x):
        """Normalizes a real valued array so the cumulative sum is unity"""
        return x / torch.sum(x)

    output = norm(output)
    target = norm(target)

    return torch.sum(torch.abs(output - target))
    # return torch.sum(torch.abs(output - target)) / torch.sum(output)


def set_learning_rate(optimizer, lr):
    def set_opt_param(optimizer, key, value):
        for group in optimizer.param_groups:
            group[key] = value

    print('[i] set learning rate to {}'.format(lr))
    set_opt_param(optimizer, 'lr', lr)
#
# def setup_propagators(sim_args):
#     prop1 = PropagatePadded(
#         nx=sim_args['grid_n'],
#         ny=sim_args['grid_n'],
#         k=sim_args['k'],
#         z_list=[sim_args['spacing_1d_to_ms1'], ],
#         dx=sim_args['dd'],
#         dy=sim_args['dd'],
#         pad_factor=1.,
#         device=device,
#     )
#     sim_args['propagator1'] = prop1
#
#     prop2 = PropagatePadded(
#         nx=sim_args['grid_n'],
#         ny=sim_args['grid_n'],
#         k=sim_args['k'],
#         z_list=[sim_args['spacing_ms2_to_detector'], ],
#         dx=sim_args['dd'],
#         dy=sim_args['dd'],
#         pad_factor=1.,
#         device=device,
#     )
#     sim_args['propagator2'] = prop2

#
# def setup_starting_fields(sim_args):
#     n_weights, pixel_width, pixel_height, pixel_spacing, end_spacing
#
#     assert isinstance(pixel_width, int)
#     assert isinstance(pixel_height, int)
#     assert isinstance(pixel_spacing, int)
#
#     n = n_weights * pixel_height + (n_weights - 1) * pixel_spacing + 2 * end_spacing  # array side length
#     field = torch.zeros(shape=(n, n, n_weights,), requires_grad=False)
#
#     for i in range(n_weights):
#         i_x = (n + 1) // 2
#         i_y = i * (pixel_spacing + pixel_height) + end_spacing + (pixel_height + 1) // 2
#         field[
#             int(i_y - pixel_height / 2.): int(i_y + pixel_height / 2.),
#             int(i_x - pixel_width / 2.): int(i_x + pixel_width / 2.),
#             i,
#         ] = 1.
#
#     return field


if __name__ == '__main__':
    device = torch.device('cuda')

    sim_args = {
        'wavelength': 1550e-9,  # simulation wavelength in meters
        'grid_n': 400,  # defines a simulation region of `grid_n` by `grid_n`
        'phase_profile_n': 400,  # defines the phase profile regions of `phase_profile_n` by `phase_profile_n`
        'grid_size': 2.0e-3,  # physical side length of the simulation grid in meters
        'input_spacing': 63.4e-6,  # period of the display inputs in meters
        'ms_to_camera_distance': 2.4e-3,  # represents the minimum distance we can experimentally place the surface

        'spacing_1d_to_ms1': 2.0e-3,
        'spacing_ms1_to_ms2': 1.5 * 1e-3,  # thickness of glass substrate
        'spacing_ms2_to_detector': 20e-3,
        'n_bins': 7,
        'n_modes': 49,
    }


    # number simulation grid pixels that represent a single output grid square
    sim_args['grid_elements_per_output'] = int(sim_args['grid_n'] / sim_args['n_modes'])
    sim_args['output_spacing'] = sim_args['grid_elements_per_output'] / sim_args['grid_n'] * sim_args['grid_size']
    sim_args['k'] = 2. * np.pi / sim_args['wavelength']
    sim_args['display_to_ms_distance'] = sim_args['ms_to_camera_distance'] * sim_args['input_spacing'] / sim_args[
        'output_spacing']
    sim_args['grid_shape'] = (sim_args['grid_n'], sim_args['grid_n'],)
    sim_args['dd'] = sim_args['grid_size'] / sim_args['grid_n']  # array element spacing

    source_args = {
        'n_weights': sim_args['n_modes'],
        'pixel_width': 2,
        'pixel_height': 2,
        'pixel_spacing': 5,
    }
    source_args['end_spacing'] = (sim_args['grid_n'] - (
            sim_args['n_modes'] * source_args['pixel_height'] + (sim_args['n_modes'] - 1) * source_args['pixel_spacing']
    )) // 2
    assert source_args['end_spacing'] >= 0, "Bounds error"

    print(f"Grid size is: {sim_args['dd'] * 1e3}mm")
    print(f"Outputs cover an range of {sim_args['n_modes'] * sim_args['output_spacing'] * 1e3}mm")
    print(f"Distance from display to metasurface is {sim_args['display_to_ms_distance'] * 1e3}mm")


    # initialize the 1D SLM basis set.
    # 1D SLM pixel coefficients
    weights = torch.ones(sim_args['n_modes'], requires_grad=False, device=device)

    best_loss = float('Inf')

    optical_layer = OpticalLayer(sim_args, source_args).to(device)

    loss_log = []  # log of loss values recorded throughout optimization
    lr = 1e-1  # initial learning rate
    optimizer = torch.optim.Adam(optical_layer.parameters(), lr=lr)

    iterations = int(5e2)  # number of iterations for optimization
    n_update = int(1e1)  # Prints information every `n_update` iterations

    torch.backends.cudnn.benchmark = True

    # Training loop
    t = time.time()
    for i in range(iterations):
        if i == int(iterations // 2):
            set_learning_rate(optimizer, lr / 2)
        if i == int(iterations / 10 * 8):
            set_learning_rate(optimizer, lr / 10)

        optimizer.zero_grad()
        # A: clean last-step gradient to ensure optimizer only uses gradient from current iteration to update params

        activation = optical_layer(weights)
        loss = loss_fn(activation, weights)

        # record the best optimized values
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_parameters = [parameter.detach().clone() for parameter in optical_layer.parameters()]
            best_psf = activation.detach().clone()

        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        if i % n_update == 0:
            t_now = time.time()
            print("Error: {}\tTimePerUpdate(s): {}\t {}/{}".format(loss, t_now - t, i + 1, iterations))
            t = t_now

        # plt.figure()
        # plt.plot(loss_log)
        # plt.show()
        print(f'Best Error: {best_loss}')

    filename = json.load(open('global_args.json'))['data_file_name']
    # save phase_profile and parameters
    pickle.dump(
        {
            "phase_profile1": phase_profile1,
            "phase_profile2": phase_profile2,
            "sim_args": sim_args,
            "source_args": source_args,
        },
        file=open(filename, "wb"),
    )

