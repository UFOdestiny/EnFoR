"""EnFoR: entailment over frame-of-reference-preserving parses.

An LLM parses each spatial statement in the frame it is stated in (no conversion); an exact solver checks entailment
over every completion the text leaves open and returns the entailed answer, or Unknown with the possible answers.
"""

__version__ = "1.0.0"
