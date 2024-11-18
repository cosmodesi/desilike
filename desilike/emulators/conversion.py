from pathlib import Path

import numpy as np
from desilike.emulators import Operation, Emulator


def convert_jaxeffort_to_desilike(fn, cls, z, params=None):

    import jaxeffort

    #quantities = ['11', 'ct', 'loop', 'st']
    ells = [0, 2, 4]

    if params is None:
        params = ['logA', 'n_s', 'h', 'omega_b', 'omega_cdm']

    def operations(params, activations, nlayers):
        operations = []
        for ilayer in range(nlayers):
            # linear network operation
            player = params['Dense_{:d}'.format(ilayer)]
            operations.append(Operation('(v[..., None, :] @ kernel)[..., 0, :] + bias', locals={name: np.asarray(player[name]) for name in ['kernel', 'bias']}))
            # non-linear activation function
            if ilayer < nlayers - 1:
                activation = activations[ilayer]
                if activation == 'silu':
                    operations.append(Operation('v / (1 + jnp.exp(-v))', locals={}))
                elif activation == 'relu':
                    operations.append(Operation('jnp.maximum(v, 0.)', locals={}))
                elif activation == 'tanh':
                    operations.append(Operation('jnp.tanh(v)', locals={}))
        return operations

    from desilike.theories import Cosmoprimo
    from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
    from desilike.emulators import _get_calculator_info

    cosmo = Cosmoprimo(fiducial='DESI', engine='class')
    cosmo.init.params['tau_reio'].update(fixed=True)
    template = DirectPowerSpectrumTemplate(cosmo=cosmo)
    calculator = cls(template=template)
    all_params = calculator.all_params

    calculator__class__, yaml_data = _get_calculator_info(calculator)

    state = {'engines': {}, 'xoperations': [],
             'yoperations': [Operation("v['11'], v['loop'], v['ct'], v['st'] = jnp.split(v.pop('pktable'), [3, 12, 16], axis=2); v", # z-axis => last and k-axis => second
                                       "v['pktable'] = jnp.moveaxis(jnp.concatenate([v.pop('11'), v.pop('loop'), v.pop('ct'), v.pop('st')], axis=-2), [0, -1], [-1, 1]); v").__getstate__()],
             'defaults': {}, 'fixed': {}}
    state.update({'varied_params': all_params.names(name=params), 'in_calculator_state': ['pktable'], 'calculator__class__': calculator__class__, 'is_calculator_sequence': False})
    state.update({'yaml_data': yaml_data, 'all_params': all_params.__getstate__()})

    def merge_operations(list_operations):
        toret = []
        for ioperation in range(len(list_operations[0])):
            _locals = {}
            for name in list_operations[0][ioperation]._locals:
                _locals[name] = np.concatenate([operations[ioperation]._locals[name][None, ...] for operations in list_operations], axis=0)
            toret.append(list_operations[0][ioperation].clone(locals=_locals))
        return toret

    for component in ['11', 'loop', 'ct', 'st']:
        model_operations, yoperations = [], []
        for iz, zz in enumerate(z):
            iz_yoperations, iz_model_operations = [], []
            for ell in ells:
                emu = jaxeffort.load_component_emulator(str(fn) + '/{:d}/{:d}/{}/'.format(iz + 1, ell, component))
                k = emu.k_grid
                iz_model_operations.append(operations(emu.NN_params['params'], emu.activations, len(emu.features)))

                limits = np.array(emu.in_MinMax)
                if 'h' in params:  # jaxeffort provides H0
                    limits[params.index('h')] = limits[params.index('h')] / 100.
                xoperations = [Operation('(v - limits[..., 0]) / (limits[..., 1] - limits[..., 0])', locals={'limits': limits})]
                limits = np.array(emu.out_MinMax).reshape(-1, len(k), 2)
                iz_yoperations.append([Operation('((v - limits[..., 0]) / (limits[..., 1] - limits[..., 0]))', inverse='v * (limits[..., 1] - limits[..., 0]) + limits[..., 0]', locals={'limits': limits})])
            model_operations.append(merge_operations(iz_model_operations))
            yoperations.append(merge_operations(iz_yoperations))
        model_operations = merge_operations(model_operations)
        yoperations = merge_operations(yoperations)

        if 'logA' in params:
            if component in ['11', 'ct']:
                yoperations.insert(0, Operation("v / (jnp.exp(X['logA']) * 1e-10)", inverse="v * jnp.exp(X['logA']) * 1e-10"))
            if component in ['loop']:
                yoperations.insert(0, Operation("v / (jnp.exp(X['logA']) * 1e-10)**2", inverse="v * (jnp.exp(X['logA']) * 1e-10)**2"))

        shape = [len(z), len(ells), 1, len(k)]
        shape[-2] = model_operations[-1]._locals['bias'].size // int(np.prod(shape))
        state['engines'][component] = {'name': 'mlp', 'params': params, 'xshape': (len(params),), 'yshape': tuple(shape),
                                       'xoperations': [operation.__getstate__() for operation in xoperations], 'yoperations': [operation.__getstate__() for operation in yoperations],
                                       'model_operations': [operation.__getstate__() for operation in model_operations], 'model_yoperations': []}
    state['fixed']['ells'] = ells
    state['fixed']['k'] = k
    state['fixed']['z'] = np.array(z)
    emulator = Emulator.from_state(state)
    return emulator


if __name__ == '__main__':

    train_dir = Path(__file__).parent / 'train'

    convert = False
    test = True
    emulator_fn = train_dir / 'reptvelocileptors' / 'emulator.npy'

    if convert:
        import jaxeffort
        base_dir = Path(jaxeffort.__file__).parent.parent
        z = np.array([2.953640434693761696e-01, 5.096288678782910919e-01, 7.057956472488681188e-01, 9.185851971138159211e-01, 9.551565018372689675e-01, 1.317065883298026430e+00, 1.490501775752700597e+00])
        from desilike.theories.galaxy_clustering.full_shape import REPTVelocileptorsPowerSpectrumMultipoles
        emulator = convert_jaxeffort_to_desilike(base_dir / 'batch_trained_velocileptors_james_effort_AP_h', REPTVelocileptorsPowerSpectrumMultipoles, z)
        emulator.save(emulator_fn)

    if test:
        import jaxeffort
        base_dir = Path(jaxeffort.__file__).parent.parent
        for ill, ell in enumerate([0, 2, 4]):
            emulator = jaxeffort.load_multipole_noise_emulator(base_dir / 'batch_trained_velocileptors_james_effort_AP_h/1/{:d}'.format(ell))
            cosmo_params = np.array([3.0363942552728806, 0.9649, 67.36, 0.02237, 0.12])  # replace with your input
            bias_params = np.array([1., 0.3, 3., 0.1, 0.2, 0.3, 0.2, 0.1, 100., 200., 300.])
            #bias_params[8:] = 0.
            ref = emulator.get_Pl(cosmo_params, bias_params)

            from matplotlib import pyplot as plt
            from desilike.emulators import EmulatedCalculator

            pt = EmulatedCalculator.load('reptvelocileptors')
            from desilike.theories.galaxy_clustering import REPTVelocileptorsTracerPowerSpectrumMultipoles
            theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(pt=pt, z=pt.z[0], prior_basis=None)
            bias_params[[2, 3]] = bias_params[[3, 2]]
            bias_params[2] = bias_params[2] + (2 / 7) * (bias_params[0] - 1.)  # bs
            bias_params[3] = 1. / 3. * (bias_params[3] - (bias_params[0] - 1.))  # b3
            bias_params[-3:] /= 1e4
            theory({name: value / 100. if name == 'h' else value for name, value in zip(['logA', 'n_s', 'h', 'omega_b', 'omega_cdm'], cosmo_params)}
                | dict(zip(['b1', 'b2', 'bs', 'b3', 'alpha0', 'alpha2', 'alpha4', 'alpha6', 'sn0', 'sn2', 'sn4'], bias_params)))
            assert np.allclose(theory.power[ill], ref)
            #theory.plot(show=True)
            #print(theory.pt.pktable[0, :, :3, 0])
