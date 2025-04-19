"""
    MLA Factory to help with picking a model
"""

def get_mla(attn_impl: str, args):
    if attn_impl == "naive":
        from .naive_mla import NaiveMLA
        return NaiveMLA(args)
    elif attn_impl == "naive+flash":
        from .naive_wflash_mla import FlashMLA
        return FlashMLA(args)
    elif attn_impl == "absorb":
        from .absorb_mla import AbsorbMLA
        return AbsorbMLA(args)
    else:
        raise ValueError(f"Unsupported attn_impl: {attn_impl}")
