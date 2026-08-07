def map_range(value, start1, stop1, start2, stop2):
    """Linear-interpolate value from [start1, stop1] to [start2, stop2].

    Mirrors Processing's map() — not clamped, extrapolates outside the range.
    """
    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1))
