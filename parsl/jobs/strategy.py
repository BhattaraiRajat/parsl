from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Dict, List, Optional, Sequence, TypedDict
import json

from parsl.executors import HighThroughputExecutor
from parsl.executors.base import ParslExecutor
from parsl.executors.status_handling import BlockProviderExecutor
from parsl.jobs.states import JobState
from parsl.process_loggers import wrap_with_logs

logger = logging.getLogger(__name__)


def read_and_remove_job_by_id(file_path, job_id):
    # Read the JSON data from the file
    scale, elasticity_type, num_nodes, nodes, start_after = None, None, None, None, None
    with open(file_path, 'r') as file:
        data = json.load(file)

    for job in data["jobs"]:
        if job["id"] == str(job_id):
            scale, elasticity_type, num_nodes, nodes, start_after = job.get("scale"), job.get(
                "elasticity_type"), job.get("num_nodes"), job.get("nodes"), job.get("start_after")

    # Modify the data by removing the entry with the specified id
    data['jobs'] = [job for job in data['jobs']
                    if int(job.get('id')) != job_id]

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    return scale, elasticity_type, num_nodes, nodes, start_after


class ExecutorState(TypedDict):
    """Strategy relevant state for an executor
    """

    idle_since: Optional[float]
    """The timestamp at which an executor became idle.
    If the executor is not idle, then None.
    """

    first: bool
    """True if this executor has not yet had a strategy poll.
    """


class Strategy:
    """Scaling strategy.

    As a workflow dag is processed by Parsl, new tasks are added and completed
    asynchronously. Parsl interfaces executors with execution providers to construct
    scalable executors to handle the variable work-load generated by the
    workflow. This component is responsible for periodically checking outstanding
    tasks and available compute capacity and trigger scaling events to match
    workflow needs.

    Here's a diagram of an executor. An executor consists of blocks, which are usually
    created by single requests to a Local Resource Manager (LRM) such as slurm,
    condor, torque, or even AWS API. The blocks could contain several task blocks
    which are separate instances on workers.


    .. code:: python

                |<--min_blocks     |<-init_blocks              max_blocks-->|
                +----------------------------------------------------------+
                |  +--------block----------+       +--------block--------+ |
     executor = |  | task          task    | ...   |    task      task   | |
                |  +-----------------------+       +---------------------+ |
                +----------------------------------------------------------+

    The relevant specification options are:
       1. min_blocks: Minimum number of blocks to maintain
       2. init_blocks: number of blocks to provision at initialization of workflow
       3. max_blocks: Maximum number of blocks that can be active due to one workflow


    .. code:: python

          active_tasks = pending_tasks + running_tasks

          Parallelism = slots / tasks
                      = [0, 1] (i.e,  0 <= p <= 1)

    For example:

    When p = 0,
         => compute with the least resources possible.
         infinite tasks are stacked per slot.

         .. code:: python

               blocks =  min_blocks           { if active_tasks = 0
                         max(min_blocks, 1)   {  else

    When p = 1,
         => compute with the most resources.
         one task is stacked per slot.

         .. code:: python

               blocks = min ( max_blocks,
                        ceil( active_tasks / slots ) )


    When p = 1/2,
         => We stack upto 2 tasks per slot before we overflow
         and request a new block


    let's say min:init:max = 0:0:4 and task_blocks=2
    Consider the following example:
    min_blocks = 0
    init_blocks = 0
    max_blocks = 4
    tasks_per_node = 2
    nodes_per_block = 1

    In the diagram, X <- task

    at 2 tasks:

    .. code:: python

        +---Block---|
        |           |
        | X      X  |
        |slot   slot|
        +-----------+

    at 5 tasks, we overflow as the capacity of a single block is fully used.

    .. code:: python

        +---Block---|       +---Block---|
        | X      X  | ----> |           |
        | X      X  |       | X         |
        |slot   slot|       |slot   slot|
        +-----------+       +-----------+

    """

    def __init__(self, *, strategy: Optional[str], policy_file: Optional[str], max_idletime: float) -> None:
        """Initialize strategy."""
        self.executors: Dict[str, ExecutorState]
        self.executors = {}
        self.max_idletime = max_idletime
        self.policy_file = policy_file

        self.strategies = {None: self._strategy_init_only,
                           'none': self._strategy_init_only,
                           'simple': self._strategy_simple,
                           'htex_auto_scale': self._strategy_htex_auto_scale,
                           'pmix_scale_simple': self._strategy_pmix_scale}

        if strategy is None:
            warnings.warn("literal None for strategy choice is deprecated. Use string 'none' instead.",
                          DeprecationWarning)

        self.strategize = self.strategies[strategy]

        logger.debug("Scaling strategy: {0}".format(strategy))

    def add_executors(self, executors: Sequence[ParslExecutor]) -> None:
        for executor in executors:
            self.executors[executor.label] = {'idle_since': None, 'first': True}

    def _strategy_init_only(self, executors: List[BlockProviderExecutor]) -> None:
        """Scale up to init_blocks at the start, then nothing more.
        """
        for executor in executors:
            if self.executors[executor.label]['first']:
                logger.debug(f"strategy_init_only: scaling out {executor.provider.init_blocks} initial blocks for {executor.label}")
                executor.scale_out_facade(executor.provider.init_blocks)
                self.executors[executor.label]['first'] = False
            else:
                logger.debug("strategy_init_only: doing nothing")

    def _strategy_simple(self, executors: List[BlockProviderExecutor]) -> None:
        self._general_strategy(executors, strategy_type='simple')

    def _strategy_htex_auto_scale(self, executors: List[BlockProviderExecutor]) -> None:
        """HTEX specific auto scaling strategy

        This strategy works only for HTEX. This strategy will scale out by
        requesting additional compute resources via the provider when the
        workload requirements exceed the provisioned capacity. The scale out
        behavior is exactly like the 'simple' strategy.

        If there are idle blocks during execution, this strategy will terminate
        those idle blocks specifically. When # of tasks >> # of blocks, HTEX places
        tasks evenly across blocks, which makes it rather difficult to ensure that
        some blocks will reach 0% utilization. Consequently, this strategy can be
        expected to scale in effectively only when # of workers, or tasks executing
        per block is close to 1.
        """
        self._general_strategy(executors, strategy_type='htex')

    def _strategy_pmix_scale(self, executors: List[BlockProviderExecutor]) -> None:
        for executor in executors:
            label = executor.label
            logger.debug(f"Strategizing for executor {label}")

            if self.executors[label]['first']:
                logger.debug(
                    f"Scaling out {executor.provider.init_blocks} initial blocks for {label}")
                executor.scale_out_facade(executor.provider.init_blocks)
                self.executors[label]['first'] = False

            # policy file induced elasticity
            job_id = executor.provider.job_id
            scale, elasticity_type, num_nodes, nodes, start_after = read_and_remove_job_by_id(
                self.policy_file, job_id)
            if elasticity_type == "manager":
                logger.info(f"Scaling Managers")
                time.sleep(int(start_after))
                if scale == "expand":
                    executor.scale_out_pmix_facade(num_nodes, nodes)
                elif scale == "shrink":
                    executor.scale_in_pmix_facade(num_nodes, nodes)
                else:
                    logger.debug(f"Error config")
            elif elasticity_type == "worker":
                logger.info(f"Scaling Workers")
                time.sleep(int(start_after))
                executor.scale_worker_pmix_facade(scale, num_nodes, nodes)
            else:
                logger.info(
                    f"Error Elasticity Type:  {elasticity_type}")

    @wrap_with_logs
    def _general_strategy(self, executors: List[BlockProviderExecutor], *, strategy_type: str) -> None:
        logger.debug(f"general strategy starting with strategy_type {strategy_type} for {len(executors)} executors")

        for executor in executors:
            label = executor.label
            logger.debug(f"Strategizing for executor {label}")

            if self.executors[label]['first']:
                logger.debug(f"Scaling out {executor.provider.init_blocks} initial blocks for {label}")
                executor.scale_out_facade(executor.provider.init_blocks)
                self.executors[label]['first'] = False

            # Tasks that are either pending completion
            active_tasks = executor.outstanding

            status = executor.status_facade

            # FIXME we need to handle case where provider does not define these
            # FIXME probably more of this logic should be moved to the provider
            min_blocks = executor.provider.min_blocks
            max_blocks = executor.provider.max_blocks
            tasks_per_node = executor.workers_per_node

            nodes_per_block = executor.provider.nodes_per_block
            parallelism = executor.provider.parallelism

            running = sum([1 for x in status.values() if x.state == JobState.RUNNING])
            pending = sum([1 for x in status.values() if x.state == JobState.PENDING])
            active_blocks = running + pending
            active_slots = active_blocks * tasks_per_node * nodes_per_block

            logger.debug(f"Slot ratio calculation: active_slots = {active_slots}, active_tasks = {active_tasks}")

            if hasattr(executor, 'connected_workers'):
                logger.debug('Executor {} has {} active tasks, {}/{} running/pending blocks, and {} connected workers'.format(
                    label, active_tasks, running, pending, executor.connected_workers))
            else:
                logger.debug('Executor {} has {} active tasks and {}/{} running/pending blocks'.format(
                    label, active_tasks, running, pending))

            # reset idle timer if executor has active tasks

            if active_tasks > 0 and self.executors[executor.label]['idle_since']:
                self.executors[executor.label]['idle_since'] = None

            # Case 1
            # No tasks.
            if active_tasks == 0:
                # Case 1a
                logger.debug("Strategy case 1: Executor has no active tasks")

                # Fewer blocks that min_blocks
                if active_blocks <= min_blocks:
                    logger.debug("Strategy case 1a: Executor has no active tasks and minimum blocks. Taking no action.")
                # Case 1b
                # More blocks than min_blocks. Scale in
                else:
                    # We want to make sure that max_idletime is reached
                    # before killing off resources
                    logger.debug(f"Strategy case 1b: Executor has no active tasks, and more ({active_blocks})"
                                 f" than minimum blocks ({min_blocks})")

                    if not self.executors[executor.label]['idle_since']:
                        logger.debug(f"Starting idle timer for executor. If idle time exceeds {self.max_idletime}s, blocks will be scaled in")
                        self.executors[executor.label]['idle_since'] = time.time()
                    idle_since = self.executors[executor.label]['idle_since']
                    assert idle_since is not None, "The `if` statement above this assert should have forced idle time to be not-None"

                    idle_duration = time.time() - idle_since
                    if idle_duration > self.max_idletime:
                        # We have resources idle for the max duration,
                        # we have to scale_in now.
                        logger.debug(f"Idle time has reached {self.max_idletime}s for executor {label}; scaling in")
                        executor.scale_in_facade(active_blocks - min_blocks)

                    else:
                        logger.debug(
                                f"Idle time {idle_duration}s is less than max_idletime {self.max_idletime}s"
                                f" for executor {label}; not scaling in")

            # Case 2
            # More tasks than the available slots.
            elif (float(active_slots) / active_tasks) < parallelism:
                logger.debug("Strategy case 2: slots are overloaded - (slot_ratio = active_slots/active_tasks) < parallelism")

                # Case 2a
                # We have the max blocks possible
                if active_blocks >= max_blocks:
                    # Ignore since we already have the max nodes
                    logger.debug(f"Strategy case 2a: active_blocks {active_blocks} >= max_blocks {max_blocks} so not scaling out")
                # Case 2b
                else:
                    logger.debug(f"Strategy case 2b: active_blocks {active_blocks} < max_blocks {max_blocks} so scaling out")
                    excess_slots = math.ceil((active_tasks * parallelism) - active_slots)
                    excess_blocks = math.ceil(float(excess_slots) / (tasks_per_node * nodes_per_block))
                    excess_blocks = min(excess_blocks, max_blocks - active_blocks)
                    logger.debug(f"Requesting {excess_blocks} more blocks")
                    executor.scale_out_facade(excess_blocks)

            elif active_slots == 0 and active_tasks > 0:
                logger.debug("Strategy case 4a: No active slots but some active tasks - could scale out by a single block")

                # Case 4a
                if active_blocks < max_blocks:
                    logger.debug("Requesting single block")

                    executor.scale_out_facade(1)
                else:
                    logger.debug("Not requesting single block, because at maxblocks already")

            # Case 4b
            # More slots than tasks
            elif active_slots > 0 and active_slots > active_tasks:
                logger.debug("Strategy case 4b: more slots than tasks")
                if strategy_type == 'htex':
                    # Scale in for htex
                    if isinstance(executor, HighThroughputExecutor):
                        if active_blocks > min_blocks:
                            excess_slots = math.ceil(active_slots - (active_tasks * parallelism))
                            excess_blocks = math.ceil(float(excess_slots) / (tasks_per_node * nodes_per_block))
                            excess_blocks = min(excess_blocks, active_blocks - min_blocks)
                            logger.debug(f"Requesting scaling in by {excess_blocks} blocks with idle time {self.max_idletime}s")
                            executor.scale_in_facade(excess_blocks, max_idletime=self.max_idletime)
                    else:
                        logger.error("This strategy does not support scaling in except for HighThroughputExecutor - taking no action")
                else:
                    logger.debug("This strategy does not support scaling in")

            # Case 3
            # tasks ~ slots
            else:
                logger.debug("Strategy case 3: no changes necessary to current block load")
