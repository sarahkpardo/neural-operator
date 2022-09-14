import torch


class LpLoss(object):
    """Computes relative and absolute L^{p} loss functions."""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0

        self.dim = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        """Computes absolute error, assuming a uniform mesh."""

        batch_size = x.shape[0]

        h = 1.0 / (x.shape[1] - 1.0)

        all_norms = (h**(self.dim/self.p))*torch.norm(
                            x.view(batch_size,-1) - y.view(batch_size,-1),
                            self.p,
                            1,) # axis

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        """Computes relative error."""

        num_examples = x.size()[0]

        diff_norms = torch.norm(
                        x.reshape(num_examples,-1) - y.reshape(num_examples,-1),
                        self.p,
                        1,) # axis
        y_norms = torch.norm(y.reshape(num_examples,-1),
                            self.p,
                            1,)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
