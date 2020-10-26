import numpy as np

from typing import Any, Tuple, Dict

import logging

from enum import Enum

import sys

class NotDescentDirection(Exception):
    pass

class ZeroDescentProduct(Exception):
    pass

class ZeroUpdate(Exception):
    pass

class Newton:

    def __init__(self, 
        obj_func : Any,
        gradient_func : Any,
        reg_inv_hessian : Any
        ):
        self.gradient_func = gradient_func
        self.obj_func = obj_func
        self.reg_inv_hessian = reg_inv_hessian

        # Logging
        handlerPrint = logging.StreamHandler()
        handlerPrint.setLevel(logging.DEBUG)
        self.log = logging.getLogger("l-bfgs")
        self.log.addHandler(handlerPrint)
        self.log.setLevel(logging.DEBUG)

        self.line_search_c = pow(10,-4)
        self.line_search_tau = 0.5

    def get_descent_inner_product(self,
        p : np.array,
        params : np.array
        ) -> float:
        grads = self.gradient_func(params)
        inner_prod = np.dot(p, grads)
        
        if inner_prod > -1e-16 and inner_prod <= 0:
            raise ZeroDescentProduct()
        elif inner_prod > 0:
            self.log.error("ERROR: Positive inner product: %.16f" % inner_prod)
            raise NotDescentDirection()
        
        return inner_prod

    def run_line_search(self, 
        p : np.array,
        params : np.array
        ) -> float:
        
        # Check inputs
        assert self.line_search_tau < 1
        assert self.line_search_tau > 0
        assert self.line_search_c > 0
        assert self.line_search_c < 1

        inner_prod = self.get_descent_inner_product(p, params)

        alpha = 1.0
        fx = self.obj_func(params)        
        fx_new = self.obj_func(params + alpha * p)
        rhs = alpha * self.line_search_c * inner_prod
        self.log.debug("   Line search armijo: obj func old: %f new: %f diff: %.16f rhs: %.16f" % (fx, fx_new, fx_new - fx, rhs))

        while fx_new - fx > rhs:
            alpha *= self.line_search_tau
            fx_new = self.obj_func(params + alpha * p)
            rhs = alpha * self.line_search_c * inner_prod

            self.log.debug("   Line search armijo: obj func old: %f new: %f diff: %.16f rhs: %.16f" % (fx, fx_new, fx_new - fx, rhs))

        return alpha

    def step(self, 
        k : int, 
        tol : float,
        params : np.array
        ) -> Tuple[bool,np.array,np.array,float]:

        update = np.zeros(len(params))
        try:

            self.log.debug("Iteration: %d [start]" % k)

            # Get current grads
            gradients = self.gradient_func(params)

            # Get regularized inv hession
            rih = self.reg_inv_hessian(params)

            # Calculate updates
            update = - np.dot(rih, gradients)
            
            # Line search
            alpha = self.run_line_search(update, params)
            update *= alpha
            self.log.debug("   Line search factor: %.16f" % alpha)

            # Commit update
            params_new = params + update

            self.log.debug("   Old params: %s" % params)
            self.log.debug("   New params: %s" % params_new)
            
            self.log.debug("Iteration: %d [finished]" % k)

            # Monitor convergence
            if np.max(abs(update)) < tol:
                raise ZeroUpdate()

            return (False, params_new, update, alpha)

        except ZeroUpdate:
            self.log.info("Converged because zero update")
            return (True, params, update, 1.0)
        except ZeroDescentProduct:
            self.log.info("Converged because zero descent inner product")
            return (True, params, update, 1.0)

    def run(self, 
        no_steps : int, 
        params_init : np.array, 
        tol : float = 1e-8, 
        store_traj : bool = False
        ) -> Tuple[bool, int, np.array, Dict[int, np.array], Dict[int, float]]:

        assert no_steps >= 1

        params = params_init.copy()

        traj = {}
        line_search = {}
        if store_traj:
            traj[0] = params.copy()

        update = np.zeros(len(params_init))
        for k in range(0,no_steps):
            converged, params, update, alpha = self.step(
                k=k,
                tol=tol,
                params=params
                )
            
            if store_traj:
                traj[k+1] = params.copy()
                line_search[k+1] = alpha

            if converged:
                return (True, k, update, traj, line_search)
        
        return (False, no_steps, update, traj, line_search)