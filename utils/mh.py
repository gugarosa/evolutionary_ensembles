from opytimizer.core.optimizer import Optimizer
from opytimizer.optimizers import abc, ba, bha, cs, fa, fpa, pso


class MetaHeuristic:
    """A Meta-Heuristic class to help users in selecting distinct meta-heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.
        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines a meta-heuristic dictionary constant with the possible values
META = dict(
    abc=MetaHeuristic(abc.ABC, dict(n_trials=10)),
    ba=MetaHeuristic(ba.BA, dict(f_min=0, f_max=2, A=0.5, r=0.5)),
    bha=MetaHeuristic(bha.BHA, dict()),
    cs=MetaHeuristic(cs.CS, dict(alpha=0.3, beta=1.5, p=0.2)),
    fa=MetaHeuristic(fa.FA, dict(alpha=0.5, beta=0.2, gamma=1.0)),
    fpa=MetaHeuristic(fpa.FPA, dict(beta=1.5, eta=0.2, p=0.8)),
    pso=MetaHeuristic(pso.PSO, dict(w=0.7, c1=1.7, c2=1.7))
)


def get_mh(name):
    """Gets a meta-heuristic by its identifier.

    Args:
        name (str): Meta-heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return META[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(
            f'Meta-heuristic {name} has not been specified yet.')
