import numpy as np


def transform_parameters(values, scales, transform_from_canonical):
    transformed_params = np.array([
        value_transform(value, scale, transform_from_canonical)
        for value, scale in zip(values, scales)
    ])

    return transformed_params


def value_transform(value, scale, transform_from_canonical):
    if scale is None:
        return value
    elif isinstance(scale, float):
        return log_transform(value, scale, transform_from_canonical)
    else:
        return bounds_transform(value, scale, transform_from_canonical)


def log_transform(value, ref_value, transform_from_canonical):
    if transform_from_canonical:
        transformed_value = ref_value * np.exp(value)
    else:
        transformed_value = np.log(value / ref_value)

    return transformed_value


def bounds_transform(value, bounds, transform_from_canonical):
    span = 0.5 * (bounds[1] - bounds[0])
    mean = 0.5 * (bounds[0] + bounds[1])

    if transform_from_canonical:
        transformed_value = span * value + mean
    else:
        clipped_value = np.clip(value, bounds[0], bounds[1])
        transformed_value = (clipped_value - mean) / span

    return transformed_value


def first_deriv_transform(value, scale):
    if isinstance(scale, float):
        return value
    else:
        return 0.5 * (scale[1] - scale[0])


def grad_transform(grad, values, scales):
    transformed_grad = np.array([
        grad_component * first_deriv_transform(value, scale)
        for grad_component, value, scale in zip(grad, values, scales)
    ])

    return transformed_grad


def get_opt_bounds(scales):
    return [get_opt_bounds_by_type(scale) for scale in scales]


def get_opt_bounds_by_type(scale):
    if isinstance(scale, float) or scale is None:
        return [None, None]
    else:
        return [-1., 1.]
