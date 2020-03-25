#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
pao.bilevel.plugins.solver7

Declare a Bender's based approach
"""

import time
import pyutilib.misc
from pyomo.core import TransformationFactory, Var, ComponentUID, Block, Objective, Set, Constraint, Any
import pyomo.opt
import pyomo.common
import pyomo.environ
from pyomo.repn import generate_standard_repn
from ..components import SubModel

@pyomo.opt.SolverFactory.register('pao.bilevel.benders',
                                  doc=\
'Solver for bilevel interdiction problems using benders cuts')
class BilevelSolver1(pyomo.opt.OptSolver):
    """
    A solver that optimizes a bilevel program interdiction problem, where
    (1) the upper objective is the opposite of the lower objective, and
    (2) the lower problem is linear and continuous.

    Only deals with a single sub-model currently.
    """

    def __init__(self, **kwds):
        kwds['type'] = 'pao.bilevel.benders'
        pyomo.opt.OptSolver.__init__(self, **kwds)
        self._metasolver = True

    def _presolve(self, *args, **kwds):
        # TODO: Override _presolve to ensure that we are passing
        #   all options to the solver (e.g., the io_options)
        self.resolve_subproblem = kwds.pop('resolve_subproblem', True)      # TODO: Change default to False
        self.use_dual_objective = kwds.pop('use_dual_objective', True)
        self._instance = args[0]
        pyomo.opt.OptSolver._presolve(self, *args, **kwds)

    def _apply_solver(self):
        start_time = time.time()

        #
        # Create a block for the no-good and benders cuts
        #
        self._instance._cuts = Block()
        self._instance._cuts.nogood = Constraint(Any)
        self._instance._cuts.benders = Constraint(Any)

        #
        # Cache an initial solution
        #
        solver = self.options.solver
        opt = pyomo.opt.SolverFactory(solver)

        eps = self.options.eps
        if not eps:
            eps = 0.01

        maxiter = self.options.maxiter
        if not maxiter:
            maxiter = 1000.

        submodel = None
        for (name, data) in self._instance.component_map(active=True).items():
            if isinstance(data, SubModel):
                for var in data._fixed:
                    var.fix(var.value)
                self._instance.reclassify_component_type(data, Block)
                submodel = data
                break

        LB = dict()
        UB = dict()
        iter = 0
        incumbent = dict()
        LB[iter-1] = float('-inf')
        UB[iter-1] = float('inf')

        while True:
            #
            # solve lower level subproblem with fixed upper level variables
            #
            results = opt.solve(submodel)
            if results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal:
                raise RuntimeError("Lower level {} didn't solve to optimality.".format(submodel.name))
            #
            # update lower bound with solution from subproblem;
            # save incumbent solution
            #
            for objective in submodel.component_objects(Objective):
                _val = pyomo.environ.value(objective)
                if _val > LB[iter-1]:
                    LB[iter] = _val
                for var in submodel._fixed:
                    incumbent[var.name] = (var.value, var)

            #
            # check if termination conditions are met
            #
            if UB - LB <= eps*LB or iter > maxiter:
                self._instance.reclassify_component_type(submodel, SubModel)
                for var in submodel._fixed:
                    var.unfix()

                #
                # Return the sub-solver return condition value and log
                #
                return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                           log=getattr(opt, '_log', None))

            #
            # Add cuts from the incumbent solution
            #
            nogood = 0.
            for idx, (val, var) in incumbent.items():
                if bool(val):
                    nogood += 1 - var
                else:
                    nogood += var
            self._instance._cuts.nogood.add_constraint(expr=nogood>=1)

            for data in submodel.component_objects(Constraint, active=True)
                for idx in data:
                    con = data[idx]
                    body_terms = generate_standard_repn(con.body, compute_values=False)
                    _flag = False
                    if body_terms.is_nonlinear():
                        for var in body_terms.nonlinear_vars:
                            if var.is_binary() and var in submodel._fixed:
                                _flag = True
                                break
                    if _flag:
                        lower_terms = generate_standard_repn(con.lower, compute_values=False) \
                            if not con.lower is None else None
                        upper_terms = generate_standard_repn(con.upper, compute_values=False) \
                            if not con.upper is None else None

        opt.solve
        with pyomo.opt.SolverFactory(solver) as opt:
            self.results = []
            #
            #
            self.results.append(opt.solve(self._instance,
                                          tee=self._tee,
                                          timelimit=self._timelimit))

            #
            # If the problem was bilinear, then reactivate the original data
            #
            if nonlinear:
                i = 0
                for v in self._instance.bilinear_data_.vlist.itervalues():
                    if abs(v.value) <= 1e-7:
                        self._instance.bilinear_data_.vlist_boolean[i] = 0
                    else:
                        self._instance.bilinear_data_.vlist_boolean[i] = 1
                    i = i + 1
                #
                self._instance.bilinear_data_.deactivate()
            if self.resolve_subproblem:
                #
                # Transform the result back into the original model
                #
                tdata = self._instance._transformation_data['pao.bilevel.linear_dual']
                unfixed_tdata = list()
                # Copy variable values and fix them
                for v in tdata.fixed:
                    if not v.fixed:
                        v.value = self._instance.find_component(v).value
                        v.fixed = True
                        unfixed_tdata.append(v)
                # Reclassify the SubModel components and resolve
                for name_ in tdata.submodel:
                    submodel = getattr(self._instance, name_)
                    submodel.activate()
                    for data in submodel.component_map(active=False).values():
                        if not isinstance(data, Var) and not isinstance(data, Set):
                            data.activate()
                    dual_submodel = getattr(self._instance, name_+'_dual')
                    dual_submodel.deactivate()
                    pyomo.common.PyomoAPIFactory('pyomo.repn.compute_standard_repn')({}, model=submodel)
                    self._instance.reclassify_component_type(name_, Block)
                    #
                    # Use the with block here so that deactivation of the
                    # solver plugin always occurs thereby avoiding memory
                    # leaks caused by plugins!
                    #
                    with pyomo.opt.SolverFactory(solver) as opt_inner:
                        #
                        # **NOTE: It would be better to override _presolve on the
                        #         base class of this solver as you might be
                        #         missing a number of keywords that were passed
                        #         into the solve method (e.g., none of the
                        #         io_options are getting relayed to the subsolver
                        #         here).
                        #
                        results = opt_inner.solve(self._instance,
                                                  tee=self._tee,
                                                  timelimit=self._timelimit)
                        print("SOLVED IN HERE2")
                        self.results.append(results)

                # Unfix variables
                for v in tdata.fixed:
                    if v in unfixed_tdata:
                        v.fixed = False

            # check that the solutions list is not empty
            if self._instance.solutions.solutions:
                self._instance.solutions.select(0, ignore_fixed_vars=True)
            #
            stop_time = time.time()
            self.wall_time = stop_time - start_time
            self.results_obj = self._setup_results_obj()
            #
            # Reactivate top level objective
            # and reclassify the submodel
            #
            for odata in self._instance.component_map(Objective).values():
                odata.activate()
            #
            # Return the sub-solver return condition value and log
            #
            return pyutilib.misc.Bunch(rc=getattr(opt, '_rc', None),
                                       log=getattr(opt, '_log', None))

    def _postsolve(self):
        #
        # Uncache the instance
        #
        self._instance = None
        #
        # Return the results object
        #
        return self.results_obj

    def _setup_results_obj(self):
        #
        # Create a results object
        #
        results = pyomo.opt.SolverResults()
        #
        # SOLVER
        #
        solv = results.solver
        solv.name = self.options.subsolver
        solv.wallclock_time = self.wall_time
        cpu_ = []
        for res in self.results:
            if not getattr(res.solver, 'cpu_time', None) is None:
                cpu_.append(res.solver.cpu_time)
        if cpu_:
            solv.cpu_time = sum(cpu_)
        #
        # TODO: detect infeasibilities, etc
        solv.termination_condition = pyomo.opt.TerminationCondition.optimal
        #
        # PROBLEM
        #
        prob = results.problem
        prob.name = self._instance.name
        prob.number_of_constraints = self._instance.statistics.number_of_constraints
        prob.number_of_variables = self._instance.statistics.number_of_variables
        prob.number_of_binary_variables = self._instance.statistics.number_of_binary_variables
        prob.number_of_integer_variables =\
            self._instance.statistics.number_of_integer_variables
        prob.number_of_continuous_variables =\
            self._instance.statistics.number_of_continuous_variables
        prob.number_of_objectives = self._instance.statistics.number_of_objectives
        #
        # SOLUTION(S)
        #
        self._instance.solutions.store_to(results)
        return results
