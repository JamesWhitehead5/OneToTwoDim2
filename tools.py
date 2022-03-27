import torch
from AngularPropTorch.angularproplayer import PropagatePadded

class OpticalLayer(torch.nn.Module):
    # Since GPU memory is limited (about 8GB), the GPU will only be used as DFT co-processor and not for storing the
    # entire network. The tensors should be large enough such that the overhead to copy between system and video memory
    # is small compared to the 2D-DFT operations

    def __init__(self, sim_args, source_args):
        super(OpticalLayer, self).__init__()

        self.sim_args = sim_args
        self.source_args = source_args

        self.register_buffer(name="starting_fields", tensor=self.generate_starting_fields(**source_args))

        assert self.starting_fields.size(1) == sim_args['grid_n'], "Change parity of grid_n"

        # initialize trainable variables
        self.phase_profile1 = torch.nn.Parameter(
            torch.rand(
                size=(sim_args['phase_profile_n'], sim_args['phase_profile_n']),
                requires_grad=True,
            ) * torch.pi * 2
        )
        self.phase_profile2 = torch.nn.Parameter(
            torch.rand(
                size=(sim_args['phase_profile_n'], sim_args['phase_profile_n']),
                requires_grad=True,
            ) * torch.pi * 2
        )

        self.prop1 = PropagatePadded(
            nx=sim_args['grid_n'],
            ny=sim_args['grid_n'],
            k=sim_args['k'],
            z_list=[sim_args['spacing_1d_to_ms1'], ],
            dx=sim_args['dd'],
            dy=sim_args['dd'],
            pad_factor=1.,
        )
        self.prop2 = PropagatePadded(
            nx=sim_args['grid_n'],
            ny=sim_args['grid_n'],
            k=sim_args['k'],
            z_list=[sim_args['spacing_ms1_to_ms2'], ],
            dx=sim_args['dd'],
            dy=sim_args['dd'],
            pad_factor=1.,
        )
        self.prop3 = PropagatePadded(
            nx=sim_args['grid_n'],
            ny=sim_args['grid_n'],
            k=sim_args['k'],
            z_list=[sim_args['spacing_ms2_to_detector'], ],
            dx=sim_args['dd'],
            dy=sim_args['dd'],
            pad_factor=1.,
        )

    def forward(self, x):
        x = x[:, None, None]
        fields = self.starting_fields * x
        fields = self.prop1.forward(fields)
        fields *= self.phase_profile1
        fields = self.prop2.forward(fields)
        fields *= self.phase_profile2
        fields = self.prop3.forward(fields)

        activations = self.bin_intensity_circular(
            intensity=torch.abs(fields)**2, n_bins=self.sim_args['n_bins'], dc=0.5, sim_args=self.sim_args,
        )

        activations = activations.flatten()

        return activations

    # @staticmethod
    # def generate_starting_fields_sparse(n_modes, pixel_width, pixel_height, pixel_spacing, end_spacing):
    #     with torch.no_grad():
    #         n = n_modes * pixel_height + (n_modes - 1) * pixel_spacing + 2 * end_spacing  # array side length
    #
    #         size = (n_modes, n, n)
    #         values = torch.ones(size=[pixel_width * pixel_height * n_modes],)
    #         i_modes = torch.kron(torch.arange(n_modes), torch.ones(size=[pixel_height * pixel_width]))
    #         i_x = (torch.arange(pixel_width) - pixel_width // 2 + n // 2).repeat(pixel_height * n_modes)
    #
    #         i_y = torch.zeros(size=values.size())
    #
    #         y_pixel_centers = torch.arange(n_modes) * pixel_spacing + end_spacing
    #         y_single_pixel = torch.kron(torch.arange(pixel_height), torch.ones(size=[pixel_width]))
    #
    #
    #         indicies = torch.stack([i_modes, i_x, i_y])
    #         sparse_tensor = torch.sparse_coo_tensor(indices=indicies, values=values, size=size, requires_grad=False)
    #         dense_tensor = sparse_tensor.to_dense()
    #         return dense_tensor
    # @staticmethod
    # def generate_starting_fields_sparse(n_modes, pixel_width, pixel_height, pixel_spacing, end_spacing):
    #     with torch.no_grad():
    #         n = n_modes * pixel_height + (n_modes - 1) * pixel_spacing + 2 * end_spacing  # array side length
    #
    #         size = (n_modes, n, n)
    #         values = torch.ones(size=[pixel_width * pixel_height * n_modes],)
    #         i_modes = torch.kron(torch.arange(n_modes), torch.ones(size=[pixel_height * pixel_width]))
    #         i_x = (torch.arange(pixel_width) - pixel_width // 2 + n // 2).repeat(pixel_height * n_modes)
    #
    #         i_y = torch.zeros(size=values.size())
    #
    #         y_pixel_centers = torch.arange(n_modes) * pixel_spacing + end_spacing
    #         y_single_pixel = torch.kron(torch.arange(pixel_height), torch.ones(size=[pixel_width]))
    #
    #
    #         indicies = torch.stack([i_modes, i_x, i_y])
    #         sparse_tensor = torch.sparse_coo_tensor(indices=indicies, values=values, size=size, requires_grad=False)
    #         dense_tensor = sparse_tensor.to_dense()
    #         return dense_tensor

    @staticmethod
    def generate_starting_fields(n_modes, pixel_width, pixel_height, pixel_spacing, end_spacing):
        n = n_modes * pixel_height + (n_modes - 1) * pixel_spacing + 2 * end_spacing  # array side length
        field = torch.zeros(size=(n_modes, n, n), requires_grad=False)

        for i in range(n_modes):
            i_x = (n + 1) // 2
            i_y = i * (pixel_spacing + pixel_height) + end_spacing + (pixel_height + 1) // 2
            field[
            i,
            int(i_y - pixel_height / 2.): int(i_y + pixel_height / 2.),
            int(i_x - pixel_width / 2.): int(i_x + pixel_width / 2.),
            ] = 1.



        return field

    @staticmethod
    def bin_intensity_circular(intensity, n_bins, dc, sim_args):
        """
        Bins the output into a grid of circles

        :param intensity: Real tensor
        :param n_bins: Grid is formed by a n_bins x n_bins grid
        :param dc: Duty cycle size of the circle in terms of period
        :param sim_args:
        :return:
        """
        device = intensity.device
        binned_output = torch.zeros(size=(n_bins, n_bins)).to(device)

        field_center = sim_args['grid_n'] // 2
        bin_centers = (field_center + sim_args['grid_elements_per_output'] * (
                    torch.arange(n_bins) - (n_bins - 1) / 2)).type(torch.int)

        assert torch.all(bin_centers - sim_args['grid_elements_per_output'] // 2 >= 0) and torch.all(
            bin_centers + sim_args['grid_elements_per_output'] // 2 < sim_args[
                'grid_n']), "Too many bins to for output size"

        ix = torch.arange(sim_args['grid_elements_per_output']) - (sim_args['grid_elements_per_output'] - 1) / 2.
        ixx, iyy = torch.meshgrid(ix, ix, indexing='xy')

        circular_mask = ixx ** 2 + iyy ** 2 < (dc * sim_args['grid_elements_per_output'] ** 2 / 4)
        # circular_mask = torch.tensor(circular_mask, device=device)

        for i, x_center in enumerate(bin_centers):
            for j, y_center in enumerate(bin_centers):
                block = intensity[
                        i + j * n_bins,
                        x_center - sim_args['grid_elements_per_output'] // 2: x_center + sim_args[
                            'grid_elements_per_output'] // 2,
                        y_center - sim_args['grid_elements_per_output'] // 2: y_center + sim_args[
                            'grid_elements_per_output'] // 2,
                        ]

                binned_output[i, j] = torch.sum(
                    block.where(condition=circular_mask, other=torch.tensor(0., dtype=block.dtype)))
                # binned_output[i, j] = torch.sum(block.where(condition=circular_mask, other=torch.zeros(size=(1, ), dtype=torch.double)))

        return binned_output


if __name__ == "__main__":
    # source_args = {'n_modes': 49, 'pixel_width': 2, 'pixel_height': 2, 'pixel_spacing': 5, 'end_spacing': 31}
    # s1 = OpticalLayer.generate_starting_fields(**source_args)
    # s2 = OpticalLayer.generate_starting_fields_sparse(**source_args)
    # print(torch.sum(torch.abs(s1 - s2)))
    pass
