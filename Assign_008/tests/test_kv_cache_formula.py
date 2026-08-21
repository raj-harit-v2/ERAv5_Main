from src.attention.gqa import kv_reduction, layout_name
from src.fertility import effective_content
from src.kv_cache_math import kv_cache_bytes, yardstick_one_user_32k
from src.pipeline_user_sentence import UserSentencePipeline
import config


def test_kv_bytes_yardstick():
    expected = 2 * 48 * 8 * 128 * 32768 * 1 * 2
    assert yardstick_one_user_32k() == expected
    assert (
        kv_cache_bytes(layers=48, h_kv=8, d_head=128, t=32768, batch=8, p_b=2)
        == expected * 8
    )


def test_gqa_layout():
    assert layout_name(8, 8) == "MHA"
    assert layout_name(8, 2) == "GQA"
    assert layout_name(8, 1) == "MQA"
    assert kv_reduction(8, 2) == 4.0


def test_fertility_teaching_ratio():
    assert effective_content(256_000, 3.0) == 256_000 / 3.0


def test_pipeline_shapes():
    pipe = UserSentencePipeline()
    pipe.eval()
    r = pipe.forward_sentence(config.EXAMPLE_SENTENCE)
    assert r.hidden.shape == (1, config.SEQ_LEN, config.D_MODEL)
    assert r.attn_out.shape == (1, config.SEQ_LEN, config.D_MODEL)
    assert r.logits.shape == (1, config.SEQ_LEN, config.VOCAB_SIZE)
    assert len(r.steps) >= 5
