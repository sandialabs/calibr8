"""Unit tests for inverse-deck parameter extraction (input_file_io).

setup_opt_parameters turns a parsed YAML deck into the optimizer's
(names, scales, block indices, initial canonical point, bounds). These cover the
extraction for both deck shapes (single block and a 'problems' block) and the
appending of text parameters.
"""
import numpy as np
import pytest

from calibr8.util.input_file_io import setup_opt_parameters


def _single_block_deck():
    # inverse block calibrates a subset (E, K, Y); initial values come from the
    # local-residual materials. E/Y use [lb, ub] scaling, K uses log scaling.
    return {
        "my problem": {
            "residuals": {
                "local residual": {
                    "materials": {
                        "body": {
                            "E": 1100.0, "nu": 0.25, "K": 200.0,
                            "Y": 10.5, "S": 0.0, "D": 0.0,
                        }
                    }
                }
            },
            "inverse": {
                "materials": {
                    "body": {
                        "E": [800.0, 1200.0],
                        "K": 100.0,
                        "Y": [9.0, 11.0],
                    }
                }
            },
        }
    }


NO_TEXT_PARAMS = (np.empty(0), [])


def test_single_block_extraction():
    names, scales, block_idx, x0, bounds = setup_opt_parameters(
        _single_block_deck(), NO_TEXT_PARAMS
    )

    assert names == ["E", "K", "Y"]
    assert scales == [[800.0, 1200.0], 100.0, [9.0, 11.0]]
    assert block_idx == [0, 0, 0]

    # E: (1100-1000)/200 = 0.5 ; K: log(200/100) ; Y: (10.5-10)/1 = 0.5
    assert x0 == pytest.approx([0.5, np.log(2.0), 0.5])
    assert bounds == [[-1.0, 1.0], [None, None], [-1.0, 1.0]]


def test_text_params_are_appended():
    text_init = np.array([3.0])
    text_scales = [None]   # identity (no scaling)
    names, scales, block_idx, x0, bounds = setup_opt_parameters(
        _single_block_deck(), (text_init, text_scales)
    )

    assert names == ["E", "K", "Y", "p_0"]
    assert scales[-1] is None
    assert bounds[-1] == [None, None]
    # block indices are intentionally not extended for text params
    assert block_idx == [0, 0, 0]
    # None scale => canonical value equals the physical value
    assert x0[-1] == pytest.approx(3.0)


def test_problems_block_takes_initial_from_first_problem():
    deck = {
        "p": {
            "problems": {
                "prob0": {"residuals": {"local residual": {"materials": {"body": {"Y": 10.5}}}}},
                "prob1": {"residuals": {"local residual": {"materials": {"body": {"Y": 99.0}}}}},
            },
            "inverse": {"materials": {"body": {"Y": [9.0, 11.0]}}},
        }
    }
    names, scales, block_idx, x0, bounds = setup_opt_parameters(deck, NO_TEXT_PARAMS)

    assert names == ["Y"]
    # initial value is taken from the first problem block (10.5 -> 0.5), not prob1
    assert x0 == pytest.approx([0.5])
