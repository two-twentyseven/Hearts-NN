from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Hyperparameters governing Hearts rule variants."""

    use_jack_of_diamonds: bool = False
    target_score: int = 100
    seed: int | None = None
