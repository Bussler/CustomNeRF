class EarlyStopping:
    """Early stopping helper based on fitness criterion.
    NeRF can be prone to local minima, in which training will quickly stall and produce blank outputs.
    Restart the training when learning stalls, if necessary.
    """

    def __init__(self, patience: int = 30, margin: float = 1e-4):
        self.best_fitness = 0.0  # Best fitness value encountered in training.In default case PSNR
        self.best_iter = 0  # iteration with best fitness value
        self.margin = margin  # min margin to be met in order to see fitness improvement
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop

    def __call__(self, iter: int, fitness: float) -> bool:
        """Check if criterion for stopping is met.

        Args:
            iter (int): current training iteration
            fitness (float): improvement metric of current training, should always improve during training

        Returns:
            bool: whether to stop the training
        """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop
