import numpy as np

import pycompadre


def moving_least_squares(source_coords, source_fields, target_coords,
        poly_order=1, epsilon_multiplier=1.6):

    dimension, = source_coords.shape[1]
    assert target_coords.shape[1] == dimension
    _, num_fields, num_steps = source_coords.shape

    num_target_points = target_coords.shape[0]
    target_fields = np.zeros((num_target_points, num_fields, num_steps))

    kp = pycompadre.KokkosParser()

    gmls_obj = pycompadre.GMLS(poly_order, dimension)

    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    gmls_helper.generateKDTree(source_coords)
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(target_coords,
        poly_order, dimension, epsilon_multiplier)

    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    gmls_obj.generateAlphas(number_of_batches=1, keep_coefficients=False)

    for step in range(num_steps):
        for idx in range(num_fields):
            target_fields[:, idx, step] = \
                gmls_helper.applyStencil(source_fields[:, idx, step],
                pycompadre.TargetOperation.ScalarPointEvaluation)

    # clean up objects in order of dependency
    del gmls_obj
    del gmls_helper
    del kp

    return target_fields
