import numpy as np
import pytest

from ndforge_worker import imatrixmoe, mixedquant


def test_the_expert_rows_of_the_kept_experts_stay_in_their_new_order():
    n, width = 6, 3
    sums = np.arange(n * width, dtype=np.float32)            # expert e holds e*3 .. e*3+2
    counts = np.arange(n, dtype=np.float32) * 10
    arrays = {"blk.2.ffn_down_exps.weight.in_sum2": sums, "blk.2.ffn_down_exps.weight.counts": counts, "blk.2.attn_q.weight.in_sum2": np.ones(4, dtype=np.float32)}
    out = imatrixmoe.slice_experts(arrays, {2: [4, 1, 5]}, n)
    assert out["blk.2.ffn_down_exps.weight.in_sum2"].tolist() == [12, 13, 14, 3, 4, 5, 15, 16, 17]
    assert out["blk.2.ffn_down_exps.weight.counts"].tolist() == [40, 10, 50]
    assert out["blk.2.attn_q.weight.in_sum2"] is arrays["blk.2.attn_q.weight.in_sum2"]


def test_a_matrix_that_does_not_fit_is_refused_in_plain_words():
    a = {"blk.0.ffn_gate_up_exps.weight.counts": np.zeros(5, dtype=np.float32)}
    with pytest.raises(imatrixmoe.MoeMatrixError, match="one per expert"):
        imatrixmoe.slice_experts(a, {0: [0]}, 6)
    with pytest.raises(imatrixmoe.MoeMatrixError, match="says nothing"):
        imatrixmoe.slice_experts({"blk.1.ffn_up_exps.weight.counts": np.zeros(6)}, {0: [0]}, 6)
    with pytest.raises(imatrixmoe.MoeMatrixError, match="outside"):
        imatrixmoe.slice_experts({"blk.0.ffn_up_exps.weight.counts": np.zeros(6)}, {0: [9]}, 6)


def test_the_fused_gate_up_experts_have_a_quant_role():
    roles = mixedquant.role_names("inkling2")
    assert "ffn_gate_up_exps" in roles["ffn_updown"] and "ffn_down_exps" in roles["ffn_down"]
