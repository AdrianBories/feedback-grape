import operator
from tracemalloc import stop
import jax
import numpy as np
from inspect import signature # Use inspect.signature to determine the number of parameters in the provided callable.
import flax.linen as nn
import jax.numpy as jnp
from pytest import param
from .fidelity import is_positive_semi_definite
# ruff: noqa N8

jax.config.update("jax_enable_x64", True)


# Answer: add in docs an example of how they can construct their own `Network to use it.`
# --> the example E nn is suitable enough to show how to use it
# Answer: make all these functions private? or just not include them in the docs? --> just not include them in the docs
class RNN(nn.Module):
    hidden_size: int  # number of features in the hidden state
    output_size: int  # number of features in the output ( 2 in the case of gamma and beta)

    @nn.compact
    def __call__(self, measurement, hidden_state):
        gru_cell = nn.GRUCell(features=self.hidden_size)

        if measurement.ndim == 1:
            measurement = measurement.reshape(1, -1)
        new_hidden_state, _ = gru_cell(hidden_state, measurement)
        # this returns the params after linear regression through the hidden state which contains
        # the information of the previous time steps and this is optimized to output best params
        # new_hidden_state = nn.Dense(features=self.hidden_size)(new_hidden_state)
        output = nn.Dense(
            features=self.output_size,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.constant(jnp.pi),
        )(new_hidden_state)
        output = nn.relu(output)
        # output = jnp.asarray(output)
        return output[0], new_hidden_state


def clip_params(params, gate_param_constraints):
    """
    Clip the parameters to be within the specified constraints. if the parameters are within the bounds, they remain unchanged.
    If they are outside the bounds, they are mapped to the bounds using a sigmoid function.

    Args:
        params: Parameters to be clipped.
        param_constraints: List of tuples specifying (min, max) for each parameter.

    Returns:
        Clipped parameters.
    """
    if gate_param_constraints == []:
        return params

    mapped_params = []
    for i, param in enumerate(params):
        min_val, max_val = gate_param_constraints[i]
        within_bounds = (param >= min_val) & (param <= max_val)

        # If within bounds, keep original; otherwise apply sigmoid mapping
        sigmoid_mapped = min_val + (max_val - min_val) * jax.nn.sigmoid(param)
        mapped_param = jnp.where(within_bounds, param, sigmoid_mapped)
        mapped_params.append(mapped_param)

    return jnp.array(mapped_params)


def apply_gate(rho_cav, gate, params, evo_type, gate_param_constraints):
    """
    Apply a gate to the given state. This also clips the parameters
    to be within the specified constraints specified by the user.

    Args:
        rho_cav: Density matrix of the cavity.
        gate: The gate function to apply.
        params: Parameters for the gate.
        evo_type: Evolution type, either "density" or "state".
        gate_param_constraints: Constraints for the parameters.

    Returns:
        tuple: Updated state.
    """
    # For non-measurement gates, apply the gate without measurement
    params = clip_params(params, gate_param_constraints)
    operator = gate(*[params])
    if evo_type == "density":
        rho_meas = operator @ rho_cav @ operator.conj().T
    else:
        rho_meas = operator @ rho_cav
    return rho_meas


def apply_channel(rho_cav, channel, params, evo_type, gate_param_constraints):
    """
    Apply a quantum channel to the given state. This also clips the parameters
    to be within the specified constraints specified by the user.

    Args:
        rho_cav: Density matrix or state vector of the cavity.
        channel: The quantum channel function to apply.
        params: Parameters for the quantum channel.
        evo_type: Evolution type, either "density" or "state".
        gate_param_constraints: Constraints for the parameters.

    Returns:
        tuple: Updated state.
    """
    # For non-measurement gates, apply the gate without measurement
    params = clip_params(params, gate_param_constraints)
    return channel(rho_cav, *[params])


def convert_to_index(measurement_history, memory_depth):
    """

    Convert measurement history from [1, -1, ...] to [0, 1, ...] and then to an integer index

    Args:
        measurement_history: List of measurements, where 1 indicates a positive measurement and -1 indicates
                             a negative measurement.
    Returns:
        int: Integer index representing the measurement history for accessing the lut.

    """
    binary_history = jnp.where(jnp.array(measurement_history) == 1, 0, 1)
    # Convert binary list to integer index (e.g., [0,1] -> 1)
    reversed_binary = binary_history[::-1]
    int_index = jnp.sum(
        ((2 ** jnp.arange(len(reversed_binary))) * reversed_binary)[:memory_depth]
    )
    return int_index


def extract_from_lut(lut, measurement_history):
    """
    Extract parameters from the lookup table based on the measurement history.

    Args:
        lut: Lookup table for parameters.
        measurement_history: History of measurements.

    Returns:
        Extracted parameters.
    """
    sub_array_idx = min(len(measurement_history) - 1, len(lut) - 1)
    sub_array_param_idx = convert_to_index(measurement_history, len(lut))
    return jnp.array(lut)[sub_array_idx][sub_array_param_idx]


def extract_from_operator_lut(lut_operators, measurement_history):
    """
    Extract operators from the lookup table based on the measurement history.

    Args:
        lut_operators: Lookup table for operators.
        measurement_history: History of measurements.

    Returns:
        Extracted operators.
    """
    sub_array_idx = min(len(measurement_history) - 1, len(lut_operators) - 1)
    sub_array_param_idx = convert_to_index(measurement_history, len(lut_operators))
    lut_operators[sub_array_idx]
    return lut_operators[sub_array_idx][sub_array_param_idx]


def reshape_params(param_shapes, flattened_params):
    """
    Reshape the parameters for the gates.
    """
    # Reshape the flattened parameters from RNN output according
    # to each gate corressponding params
    reshaped_params = []
    param_idx = 0
    for shape in param_shapes:
        num_params = int(np.prod(shape))
        # rnn outputs a flat list, this takes each and assigns according to the shape
        gate_params = flattened_params[
            param_idx : param_idx + num_params
        ].reshape(shape)
        reshaped_params.append(gate_params)
        param_idx += num_params

    new_params = reshaped_params
    return new_params


def prepare_parameters_from_dict(params_dict):
    """
    Convert a nested dictionary of parameters to a flat list and record shapes.

    Args:
        params_dict: Nested dictionary of parameters.

    Returns:
        tuple: Flattened parameters list and list of shapes.
    """
    res = []
    shapes = []
    for value in params_dict.values():
        flat_params = jax.tree_util.tree_leaves(value)
        res.append(jnp.array(flat_params, dtype=jnp.float64))
        shapes.append(jnp.array(flat_params).shape[0])
    return res, shapes


def construct_ragged_row(
    num_of_rows, num_of_columns, param_constraints, init_flat_params, rng_key
):
    """
    Construct a ragged row of parameters for the gates in the lookup table.

    Args:
        num_of_rows: Number of rows in this array which would be a ragged row in the lut before padding.
        num_of_columns: Number of columns of the array (the total number of parameters of the system).
        param_constraints: List of tuples specifying (min, max) for each parameter. If not specfied == [] and then
            the initial flat parameters are used for all rows.
        init_flat_params: Initial flat parameters for the gates.
        rng_key: JAX random key for random parameter initialization.

    Returns:
        One ragged row of the lookup table with the specified number of rows and columns.

    """
    res = []
    if len(param_constraints) == 0:
        for i in range(num_of_rows):
            flattened = jnp.concatenate([arr for arr in init_flat_params])
            res.append(flattened)
        return res
    else:
        for i in range(num_of_rows):
            row = []
            for j in range(num_of_columns):
                rng_key, subkey = jax.random.split(rng_key)
                val = jax.random.uniform(
                    subkey,
                    shape=(),
                    minval=param_constraints[j][0],
                    maxval=param_constraints[j][1],
                )
                row.append(val)
            res.append(jnp.array(row))
        return res


def convert_system_params(system_params):
    """
    Convert system_params format to (initial_params, parameterized_gates, measurement_indices, param_constraints, c_ops, decay_indices) format.

    Args:
        system_params: List of NamedTuples. Either Gate or Decay NamedTuples.

    Returns:
        tuple:
            - initial_params: dict mapping gate names/types to parameter lists
            - parameterized_gates: list of gate functions
            - measurement_indices: list of indices where measurement gates appear
            - param_constraints: list of parameter constraints for each gate
            - c_ops: list of collapse operators for decay gates
            - decay_indices: list of indices where decay gates appear
    """
    initial_params = {}
    parameterized_gates = []
    measurement_indices = []
    param_constraints = []
    c_ops = []
    decay_indices = []
    channel_indices = []

    def _Gate_validity_checks(gate):
        """
        Checks if the provided gate is a valid unitary or POVM element by evaluating it at the initial parameters.
        """
        if not gate.measurement_flag and not gate.quantum_channel_flag: # Check if gate is callable on initial parameters and unitary
            unitary = gate.gate(gate.initial_params)
            
            if len(unitary.shape) == 2:

                assert unitary.shape[0] == unitary.shape[1], "The provided gate is not a square matrix."
                identity = jnp.eye(unitary.shape[0])
                if not jnp.allclose(unitary @ unitary.conj().T, identity):
                    if jnp.allclose(unitary, unitary.conj().T):
                        raise ValueError("The provided gate is not unitary but Hermitian. Did you perhaps provide a Hamiltonian instead of a unitary?")
                    else:
                        raise ValueError("The provided gate is not unitary. Did you perhaps mistake jnp.exp for a matrix exponential jax.scipy.linalg.expm?")
                
            elif len(unitary.shape) == 0:
                assert jnp.isclose(jnp.linalg.norm(unitary), 1.0), f"The provided gate is not a normalized state. Instead it is a scalar of value {unitary}."

            else:
                raise ValueError("The provided gate must be either a unitary matrix or 1.")

        elif gate.measurement_flag: # Check if gate is callable on initial parameters and a valid POVM element
            assert gate.quantum_channel_flag == False, "A gate cannot be both a measurement and a quantum channel."

            # Use inspect.signature to determine the number of parameters in the provided callable.
            sig = signature(gate.gate)
            if len(sig.parameters) != 2:
                raise ValueError(
                    "The Positive operator valued measure gate you supplied must have two arguments. "
                    "The first argument is the measurement outcome (1, or -1) and the second argument is the list "
                    "of optimizable parameters for the measurement gate."
                )

            M_0 = gate.gate(-1, gate.initial_params)
            M_1 = gate.gate(1, gate.initial_params)
            
            for M in [M_0, M_1]:
                assert M.shape[0] == M.shape[1], "The provided measurement operator must be a square matrix."

                # redundant: E = M.conj().T @ M >= 0 by construction
                # E = M.conj().T @ M
                # assert is_positive_semi_definite(E), "The provided measurement operator M does not satisfy M^† M >= 0 (positive semi-definite)."

            assert jnp.allclose(M_0.conj().T @ M_0 + M_1.conj().T @ M_1, jnp.eye(M_0.shape[0])), "The provided measurement operators do not sum to the identity."
        else: # Quantum channel
            # Use inspect.signature to validate quantum channel callables
            sig = signature(gate.gate)
            if len(sig.parameters) != 2:
                raise ValueError(
                    "The quantum channel gate you supplied must have two arguments. "
                    "The first argument is the density matrix / state vector and the second argument is the list "
                    "of optimizable parameters for the quantum channel."
                )

            # Check if the quantum channel works on a state vector or density matrix is not implemented yet

    for i, gate_config in enumerate(system_params):
        if hasattr(gate_config, "c_ops"):
            c_ops.append(gate_config.c_ops)
            decay_indices.append(i)
        else:
            _Gate_validity_checks(gate_config) # Validate the gate

            gate_func = gate_config.gate
            if isinstance(gate_config.initial_params, jnp.ndarray):
                # If initial_params is a numpy array, convert it to a list
                params = gate_config.initial_params.tolist()
            else:
                params = gate_config.initial_params
            is_measurement = gate_config.measurement_flag
            is_channel = gate_config.quantum_channel_flag

            # Add gate to parameterized_gates list
            parameterized_gates.append(gate_func)

            # If this is a measurement gate or quantum channel, add its index
            if is_measurement:
                measurement_indices.append(i)
            elif is_channel:
                channel_indices.append(i)

            param_name = f"gate_{i}"

            initial_params[param_name] = params

            # Add parameter constraints if provided
            if gate_config.param_constraints is not None:
                param_constraints.append(gate_config.param_constraints)

            if len(param_constraints) > 0 and (
                len(param_constraints) != len(parameterized_gates)
            ):
                raise TypeError(
                    "If you provide parameter constraints for some gates, you need to provide them for all gates."
                )

    return (
        initial_params,
        parameterized_gates,
        measurement_indices,
        param_constraints,
        c_ops,
        decay_indices,
        channel_indices,
    )


def get_trainable_parameters_for_no_meas(
    initial_parameters, param_constraints, num_time_steps, rng_key
):
    """

    This function prepares the trainable parameters for the case for which no measurement is
    performed. Meaning this is just for normal gate-parameterized GRAPE optimization.

    User enters the initial parameters and if they want to constrain the parameters
    to be within a certain range, they can specify the `param_constraints` argument.

    if that is not provided the initial parameters are used for all time steps.

    Args:
        initial_parameters: Initial parameters for the gates.
        param_constraints: List of tuples specifying (min, max) for each parameter.
        num_time_steps: Number of time steps for the optimization. (used to infer the dimension of the trainable parameters)
        rng_key: JAX random key for random parameter initialization between specified bounds if param_constraints is provided.

    Returns:
        List of trainable parameters for each time step.


    """
    trainable_params = []
    flat_params, _ = prepare_parameters_from_dict(initial_parameters)
    trainable_params.append(flat_params)
    for i in range(num_time_steps - 1):
        gate_params_list = []
        if param_constraints != []:
            for gate_constraints in param_constraints:
                sampled_params = []
                for var_bounds in gate_constraints:
                    rng_key, subkey = jax.random.split(rng_key)
                    var = jax.random.uniform(
                        subkey,
                        shape=(),
                        minval=var_bounds[0],
                        maxval=var_bounds[1],
                    )
                    sampled_params.append(var)
                gate_params_list.append(jnp.array(sampled_params))
            trainable_params.append(gate_params_list)
        else:
            # if no parameter constraints are provided, we just use the initial parameters
            # for all time steps as initial parameters
            trainable_params.append(flat_params)

    return trainable_params


def _evaluate_params(params, parameterized_gates, decay_indices, measurement_indices, channel_indices, param_constraints, msmt_idx=0, force_evaluate_all=True, operator_shapes_prev=None):
        lut_operators_item = []
        decay_count_so_far = 0
        operator_shapes = []

        start_idx = 0
        stop_idx = len(parameterized_gates) + len(decay_indices)

        if not force_evaluate_all:
            if msmt_idx > 0:
                start_idx = measurement_indices[msmt_idx - 1] + 1 # evaluate from the gate after the last measurement
            if msmt_idx < len(measurement_indices):
                stop_idx = measurement_indices[msmt_idx] + 1 # evaluate up to and including the next measurement

        assert start_idx == 0 or operator_shapes_prev is not None, "For start_idx>0, operator_shapes_prev must be provided to reshape the parameters correctly."
        assert stop_idx == len(parameterized_gates) + len(decay_indices) or operator_shapes is not None, "For stop_idx < len(parameterized_gates) + len(decay_indices), operator_shapes must be provided to reshape the parameters correctly."

        decay_count_so_far = 0
        channels_so_far = 0
        for i in range(0, start_idx):
            if i in decay_indices:
                decay_count_so_far += 1 # skip as it is not parametrized
            elif i in channel_indices:
                channels_so_far += 1 # skip as it can not be represented in the LUT
            else:
                shape = operator_shapes_prev[i - decay_count_so_far - channels_so_far]
                lut_operators_item.append(jnp.zeros(shape))  # Placeholder for operators which will never be used
                operator_shapes.append(shape)

        # Apply each gate in sequence
        for i in range(start_idx, stop_idx):
            if i in decay_indices:
                decay_count_so_far += 1
            elif i in measurement_indices:
                povm_params = clip_params(
                    params[i - decay_count_so_far],
                    param_constraints[i - decay_count_so_far]
                    if param_constraints != []
                    else []
                )
                    
                povm_fun = parameterized_gates[i - decay_count_so_far]
                M_plus = povm_fun(1, *[povm_params])
                M_minus = povm_fun(-1, *[povm_params])
                M_plus_minus = jnp.vstack([M_plus, M_minus])
                lut_operators_item.append(M_plus_minus)

                operator_shapes.append(M_plus_minus.shape) # assumes M_plus and M_minus have the same shape
            elif i not in channel_indices:
                gate = parameterized_gates[i - decay_count_so_far]
                params = params[i - decay_count_so_far]
                params = clip_params(params,
                    param_constraints[i - decay_count_so_far]
                    if param_constraints != []
                    else []
                )
                op = gate(*[params])
                lut_operators_item.append(op)
                operator_shapes.append(op.shape)
            else:
                channels_so_far += 1 # skip as it can not be represented in the LUT

        for i in range(stop_idx, len(parameterized_gates) + len(decay_indices)):
            if i in decay_indices:
                decay_count_so_far += 1 # skip as it is not parametrized
            elif i in channel_indices:
                channels_so_far += 1 # skip as it can not be represented in the LUT
            else:
                shape = operator_shapes_prev[i - decay_count_so_far - channels_so_far]
                lut_operators_item.append(jnp.zeros(shape))  # Placeholder for operators which will never be used
                operator_shapes.append(shape)

        return lut_operators_item, operator_shapes


def evaluate_lut_params(
    initial_params,
    lut,
    parameterized_gates,
    decay_indices,
    measurement_indices,
    channel_indices,
    param_constraints,
    param_shapes,
    operator_shapes=None,
):
    """
    Evaluates all operators in the lookup table with the provided parameters, so that it does not have to be done
    repeatedly during the simulation.
    
    Returns a lookup table with the evaluated operators (jnp.ndarray).
    """
    lut_operators = []
    for col in range(len(lut)):
        lut_operators_col = []

        for j in range(2**(col + 1)): # Loop through all possible measurement histories of length col + 1
            # Update measurement history based on binary representation of j
            bit_str = format(j, f'0{col + 1}b')
            measurement_history = [1 if bit == '0' else -1 for bit in bit_str]

            # Extract parameters from LUT for the current measurement history
            extracted_lut_params = extract_from_lut(
                lut, measurement_history
            )
            extracted_lut_params = reshape_params(
                param_shapes, extracted_lut_params
            )

            lut_operators_item, _ = _evaluate_params(
                extracted_lut_params,
                parameterized_gates,
                decay_indices,
                measurement_indices,
                channel_indices,
                param_constraints,
                msmt_idx=col+1,
                operator_shapes_prev=operator_shapes,
                force_evaluate_all=col == len(lut) - 1,
            )

            # Flatten the list of operators
            lut_operators_item = jnp.concatenate([op.flatten() for op in lut_operators_item])
            lut_operators_col.append(lut_operators_item)

        lut_operators_col = jnp.pad( # pad with zeros to make all columns the same size
            jnp.array(lut_operators_col),
            ((0, 2**(len(lut)) - 2**(col + 1)), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        lut_operators.append(lut_operators_col)

    initial_operators, _ = _evaluate_params(
        initial_params,
        parameterized_gates,
        decay_indices,
        measurement_indices,
        channel_indices,
        param_constraints,
        msmt_idx=0,
        operator_shapes_prev=operator_shapes,
        force_evaluate_all=False,
    ) # initial_operators, not flattened

    return initial_operators, lut_operators
