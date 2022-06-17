import torch
from torch.optim import Optimizer



class SCRNOptimizer(Optimizer):
    def __init__(self, params, inner_itr=10, ro=100, l=100, epsilon=1e-3, c_prime=0.1, step_size=0.001):

        self.ro = ro
        self.l = l
        self.epsilon = epsilon
        self.c_prime = c_prime
        self.inner_itr = inner_itr
        self.step_size = 1 / (20 * l)
        self.iteration = -1
        defaults = dict()
        self.sqr_grads_norms = 0
        self.last_grad_norm = 0

        super(SCRNOptimizer, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['displacement'] = torch.zeros_like(p)

    def compute_norm_of_list_var(self, array_):
        """
        Args:
        param array_: list of tensors
        return:
        norm of the flattened list
        """
        norm_square = 0
        for i in range(len(array_)):
            norm_square += array_[i].norm(2).item() ** 2
        return norm_square ** 0.5

    def inner_product_of_list_var(self, array1_, array2_):

        """
        Args:
        param array1_: list of tensors
        param array2_: list of tensors
        return:
        The inner product of the flattened list
        """

        sum_list = 0
        for i in range(len(array1_)):
            sum_list += torch.sum(array1_[i] * array2_[i])
        return sum_list

    def cubic_subsolver(self, grads, param, grad_norm: float, epsilon: float, ro: float, l: float):
        """
        solve the sub problem with gradient decent
        """
        deltas = [0] * len(grads)
        g_tildas = [0] * len(grads)

        # compute the hessian
        # hessian = torch.autograd.grad(outputs=grads, inputs=param, retain_graph=True)
        # turn of unwanted actions
        with torch.no_grad():
            if grad_norm >= l ** 2 / self.ro:
                # compute hessian vector with respect to grads
                hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                          grad_outputs=grads, retain_graph=True)
                g_t_dot_bg_t = self.inner_product_of_list_var(grads, hvp) / (ro * (grad_norm ** 2))
                R_c = -g_t_dot_bg_t + (g_t_dot_bg_t ** 2 + 2 * grad_norm / ro) ** 0.5
                for i in range(len(grads)):
                    deltas[i] = -R_c * grads[i].clone() / grad_norm

            else:
                sigma = self.c_prime * (epsilon * ro) ** 0.5 / l
                for i in range(len(grads)):
                    deltas[i] = torch.zeros(grads[i].shape).to(device)
                    khi = torch.rand(grads[i].shape).to(device)
                    g_tildas[i] = grads[i].clone() + sigma * khi
                for t in range(self.inner_itr):
                    # compute hessian vector with respect to delta
                    hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                              grad_outputs=deltas, retain_graph=True)
                    deltas_norm = self.compute_norm_of_list_var(deltas)
                    if self.compute_norm_of_list_var(hvp)>200:
                        break

                    for i in range(len(grads)):
                        deltas[i] = deltas[i] - self.step_size * (
                                g_tildas[i] + hvp[i] + ro / 2 * deltas_norm * deltas[i])


        # compute hessian vector with respect to delta
        hvp = torch.autograd.grad(outputs=grads, inputs=param,
                                  grad_outputs=deltas, retain_graph=True)
        deltas_norm = self.compute_norm_of_list_var(deltas)
        delta_m = 0
        for i in range(len(grads)):
            delta_m += torch.sum(grads[i] * deltas[i]) + 0.5 * torch.sum(deltas[i] * hvp[i]) + ro / 6 * deltas_norm ** 3

        deltas_norm = 0
        # update the displacement
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                state = self.state[p]
                deltas_norm += deltas[i].norm(2).item() ** 2
                state['displacement'] = deltas[i]
                i += 1

        return delta_m.item(), deltas_norm ** 0.5

    def cubic_finalsolver(self, grads, param, epsilon: float, ro: float, l: float):
        """
        solve the sub problem with gradient decent
        """
        grads_m = [0] * len(grads)
        with torch.no_grad():
            deltas = [0] * len(grads)
            for i in range(len(grads)):
                deltas[i] = torch.zeros_like(grads[i])
                grads_m[i] = grads[i].clone()
            while self.compute_norm_of_list_var(grads_m, ) > epsilon / 2:
                hvp = torch.autograd.grad(outputs=grads, inputs=param, grad_outputs=deltas, retain_graph=True)
                for i in range(len(grads)):
                    deltas[i] = deltas[i] - self.step_size * grads_m[i]
                deltas_norm = self.compute_norm_of_list_var(deltas)
                for i in range(len(grads)):
                    grads_m[i] = grads[i] + hvp[i] + ro / 2 * deltas_norm * deltas[i]

            # update the displacement
            for group in self.param_groups:
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        state = self.state[p]
                        state['displacement'] = deltas[i]
                        i += 1

    def update_parameters(self, ):

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    displacement = state['displacement']
                    p.add_(displacement.clone())

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.iteration += 1

        # compute the gradiant
        for group in self.param_groups:
            grads = []
            param = []
            grad_square_norm = 0

            for p in group['params']:
                if p.grad is None:
                    continue
                with torch.no_grad():
                    d_p = p.grad
                    grad_square_norm += d_p.norm(2).item() ** 2
                grads.append(p.grad)
                param.append(p)

            # hessian = self.eval_hessian(grads,param)
            # print(hessian.shape)
            # e, _ = torch.eig(torch.tensor(hessian))
            # lambda_min = torch.min(e[:, 0]).item()
            delta_m, deltas_norm = self.cubic_subsolver(grads, param, grad_square_norm ** 0.5, self.epsilon, self.ro,
                                                        self.l)
            # if delta_m >= -(self.epsilon ** 3 / self.ro) ** 0.5 / 100:
            #     self.cubic_finalsolver(grads, param, self.epsilon, self.ro, self.l)
            #     self.update_parameters()
            #     return loss
            # else:

            self.update_parameters()

            # with tabular.prefix("SCRN" + '/'):
            #     tabular.record('delta of m', delta_m)
            #     tabular.record('norm of gradient', grad_square_norm ** (1. / 2))
            #     tabular.record('norm of deltas', deltas_norm)
            #     # tabular.record('landa min', lambda_min)
            #     logger.log(tabular)
        return loss

    # eval Hessian matrix
    def eval_hessian(self, loss_grad, params):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], params, create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian.cpu().data.numpy()
