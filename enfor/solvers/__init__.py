"""Exact solvers: each reads a parse written in the frames of the text and performs every frame conversion itself."""


class Undecided(Exception):
    """The parse cannot be read or the solver leaves the question undecided (the caller falls back to CoT)."""
