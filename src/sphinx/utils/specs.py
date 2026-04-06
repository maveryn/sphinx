# sphinx/utils/specs.py
from dataclasses import is_dataclass, replace

def clone_spec(spec, **updates):
    """
    Make a copy of a motif spec with field updates, robust to different spec types.
    Priority:
      1) spec.clone(**updates) if available
      2) dataclasses.replace if spec is a dataclass
      3) reconstruct via __class__(**spec.__dict__)
    """
    # 1) If spec already exposes clone
    if hasattr(spec, "clone") and callable(spec.clone):
        return spec.clone(**updates)

    # 2) Dataclass path
    if is_dataclass(spec):
        try:
            return replace(spec, **updates)
        except TypeError:
            pass  # fall through to manual

    # 3) Generic reconstruction
    base = {}
    if hasattr(spec, "to_dict"):
        base = dict(spec.to_dict())
    elif hasattr(spec, "__dict__"):
        base = dict(spec.__dict__)

    # shallow copy/merge
    if "extra" in base and isinstance(base["extra"], dict):
        base["extra"] = {**base["extra"]}
    base.update(updates)
    return spec.__class__(**base)

def _prefer_asym_mode(motif, spec):
    ex = dict(getattr(spec, "extra", {}) or {})
    if ex.get("mode") != "asym":
        ex["mode"] = "asym"
        try:
            return motif.clamp_spec(spec.clone(extra=ex))
        except Exception:
            return spec.clone(extra=ex)
    return spec

def _prefer_sym_mode(motif, spec):
    ex = dict(getattr(spec, "extra", {}) or {})
    if ex.get("mode") != "sym":
        ex["mode"] = "sym"
        try:
            return motif.clamp_spec(spec.clone(extra=ex))
        except Exception:
            return spec.clone(extra=ex)
    return spec
