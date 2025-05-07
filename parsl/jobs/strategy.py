from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Dict, List, Optional, Sequence, TypedDict, Union, Tuple
import json
import os
import fcntl
import os

from parsl.launchers import PMIxLauncher, SimplePMIxLauncher
from parsl.executors import HighThroughputExecutor
from parsl.executors.base import ParslExecutor
from parsl.executors.status_handling import BlockProviderExecutor
from parsl.jobs.states import JobState
from parsl.process_loggers import wrap_with_logs

from parsl.launchers import PMIxLauncher, SimplePMIxLauncher

logger = logging.getLogger(__name__)


def read_and_remove_job_by_id(file_path: str, job_id: Union[int, str]) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[list], Optional[Union[int, float]]]:
    """Read a jobs JSON file, return fields for the given job_id, and remove that job entry.

    This is safe under concurrent access by using an exclusive file lock and
    in-place rewrite (seek -> write -> truncate -> fsync).

    Returns (scale, num_nodes, nodes, start_after) or Nones if not found.
    """
    scale = None
    num_nodes = None
    nodes = None
    start_after = None

    target_id = str(job_id)

    try:
        # Open read/write so we can update in-place under a single lock
        with open(file_path, 'r+', encoding='utf-8') as f:
            # Exclusive lock for read-modify-write
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in %s: %s", file_path, e)
                    data = {}

                jobs = data.get("jobs")
                if not isinstance(jobs, list):
                    logger.warning("No 'jobs' list in %s; nothing to remove", file_path)
                    # keep return values as None
                    return scale, num_nodes, nodes, start_after

                # Find the job and capture fields
                found_idx = None
                for idx, job in enumerate(jobs):
                    jid = str(job.get("id"))
                    if jid == target_id:
                        found_idx = idx
                        scale = job.get("scale")
                        num_nodes = job.get("num_nodes")
                        nodes = job.get("nodes")
                        start_after = job.get("start_after")
                        break

                # Remove the job if present
                if found_idx is not None:
                    del jobs[found_idx]
                    data["jobs"] = jobs
                    # Rewrite file in-place
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
                    f.flush()
                    os.fsync(f.fileno())
                else:
                    logger.info("Job id %s not found in %s; leaving file unchanged", target_id, file_path)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except FileNotFoundError:
        logger.warning("Jobs file %s not found; nothing to remove", file_path)
    except Exception as e:
        logger.error("Error updating %s: %s", file_path, e)
    logger.info("Read job id %s from %s: scale=%s num_nodes=%s nodes=%s start_after=%s", target_id, file_path, scale, num_nodes, nodes, start_after)
    return scale, num_nodes, nodes, start_after

def _get_elasticity_type_from_launcher(launcher) -> str:
    """Return 'manager' for PMIxLauncher, 'worker' for SimplePMIxLauncher,
    or try launcher.elasticity_type; otherwise 'unknown'."""
    try:
        if isinstance(launcher, PMIxLauncher):
            return "manager"
        if isinstance(launcher, SimplePMIxLauncher):
            return "worker"
        etype = getattr(launcher, "elasticity_type", None)
        if isinstance(etype, str) and etype:
            return etype.lower()
    except Exception:
        pass
    return "unknown"

def check_job_request_exists(file_path, job_id):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            job_requests_data = json.load(file)
            if any(int(job_entry["job_id"]) == int(job_id) for job_entry in job_requests_data.get("job_requests", [])):
                logger.info(f" Job Request {job_id} entry already exists.")
                return True
    return False

def check_elasticity_active(file_path, worker_change_file, job_id):
    if os.path.exists(file_path): # check if policy already exists 
        with open(file_path, "r") as file:
            policy_data = json.load(file)
            if any(job_entry["id"] == str(job_id) for job_entry in policy_data["jobs"]):
                logger.info(f"[Policy] Job {job_id} entry already exists.")
                return True
    if os.path.exists(worker_change_file) and os.path.getsize(worker_change_file) > 0: # check if worker ongoing changes
        logger.info(f"Worker change from previous elastic event ongoing.")
        return True

    return False

def update_job_requests_file(job_requests_file, scale, num_nodes, job_id):
    new_entry = {
        "job_id": str(job_id),
        "scale": scale,
        "num_nodes": num_nodes,
        "status": "pending"
    }
    if not os.path.exists(job_requests_file) or os.stat(job_requests_file).st_size == 0:
        logger.warning("Job requests file does not exist or is empty. Creating a new file.")
        job_requests_data = {"job_requests": [new_entry]}
        with open(job_requests_file, "w") as new_file:
            json.dump(job_requests_data, new_file, indent=4)
    else:
        # Read and update the policy file
        with open(job_requests_file, "r+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                job_requests_data = json.load(file)

                if not any(job_entry["job_id"] == new_entry["job_id"] for job_entry in job_requests_data.get("job_requests", [])):
                    job_requests_data["job_requests"].append(new_entry)
                    file.seek(0)
                    file.truncate()  # Clear the file before writing
                    json.dump(job_requests_data, file, indent=4)
                    logger.info(f"New job request entry added for Job {job_id}.")
                else:
                    logger.info(f"Job request {job_id} entry already exists.")

            except json.JSONDecodeError:
                logger.error("Invalid JSON format. Resetting job requests file.")
                job_requests_data = {"job_requests": [new_entry]}
                with open(job_requests_file, "w") as new_file:
                    json.dump(job_requests_data, new_file, indent=4)

            finally:
                fcntl.flock(file, fcntl.LOCK_UN)


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
        path_job_requests= os.path.dirname(policy_file)
        self.job_requests_file =  f"{path_job_requests}/job_requests.json"

        self.current_tasks_per_node = -1
        self.current_nodes_per_block = -1

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

            active_tasks = executor.outstanding
            if self.current_nodes_per_block == -1:
                self.current_nodes_per_block = executor.provider.nodes_per_block
            if self.current_tasks_per_node == -1:
                self.current_tasks_per_node = executor.workers_per_node

            job_id = executor.provider.job_id
            try:
                if not self.policy_file or not os.path.exists(self.policy_file):
                    logger.info("No policy file configured or found; skipping elasticity for job %s", job_id)
                else:
                    scale, num_nodes, nodes, start_after = read_and_remove_job_by_id(self.policy_file, job_id)
                    scale_str = str(scale).lower() if scale is not None else None
                    num_nodes = int(num_nodes) if num_nodes is not None else None
                    etype = _get_elasticity_type_from_launcher(executor.provider.launcher)
                    logger.info("Policy elasticity for job %s: type=%s scale=%s num_nodes=%s nodes=%s start_after=%s", job_id, etype, scale, num_nodes, nodes, start_after)
                    if not scale_str:
                        logger.info("No scale directive found for job %s; skipping", job_id)
                        continue
                    # normalize start_after delay
                    delay = 0.0
                    try:
                        if start_after is not None:
                            delay = max(0.0, float(start_after))
                    except Exception:
                        delay = 0.0
                    if delay:
                        time.sleep(delay)

                    if etype == "manager":
                        if scale_str == "expand":
                            if num_nodes and num_nodes > 0:
                                try:
                                    logger.info("Manager expand by %d nodes (nodes=%r) for job %s", num_nodes, nodes, job_id) 
                                    executor.scale_out_pmix_facade(num_nodes, nodes)
                                except Exception as e:
                                    logger.warning("Manager expand failed for job %s: %s", job_id, e)
                            else:
                                logger.info("Invalid num_nodes for manager expand: %r", num_nodes)
                        elif scale_str == "shrink":
                            if num_nodes and num_nodes > 0:
                                try:
                                    logger.info("Manager shrink by %d nodes (nodes=%r) for job %s", num_nodes, nodes, job_id)  
                                    executor.scale_in_pmix_facade(num_nodes, nodes)
                                except Exception as e:
                                    logger.warning("Manager shrink failed for job %s: %s", job_id, e)
                            else:
                                logger.info("Invalid num_nodes for manager shrink: %r", num_nodes)
                        else:
                            logger.debug("Unknown manager scale directive: %r", scale)
                    elif etype == "worker":
                        try:
                            logger.info("Worker scale %s by %r nodes (nodes=%r) for job %s", scale_str, num_nodes, nodes, job_id)
                            executor.scale_worker_pmix_facade(scale_str, num_nodes, nodes)
                        except Exception as e:
                            logger.warning("Worker scale %s failed for job %s: %s", scale_str, job_id, e)
                    else:
                        logger.info("Unrecognized Elasticity Type: %r", etype)
            except Exception as e:
                logger.error("Policy elasticity handling failed for job %s: %s", job_id, e)

            # scaling logic start
            if active_tasks == 0:
                logger.info("Executor has no active tasks. Verifying inactivity before canceling.")
                # We want to make sure that max_idletime is reached
                # before killing off resources

                if not self.executors[executor.label]['idle_since']:
                    logger.debug(f"Starting idle timer for executor. If idle time exceeds {self.max_idletime}s, allocation will be canceled")
                    self.executors[executor.label]['idle_since'] = time.time()
                idle_since = self.executors[executor.label]['idle_since']
                assert idle_since is not None, "The `if` statement above this assert should have forced idle time to be not-None"

                idle_duration = time.time() - idle_since
                if idle_duration > self.max_idletime:
                    # We have resources idle for the max duration,
                    # we have to cancel allocation now.
                    logger.debug(f"Idle time has reached {self.max_idletime}s for executor {label}; scaling in")

                    executor.provider.cancel([job_id])

                else:
                    logger.debug(
                            f"Idle time {idle_duration}s is less than max_idletime {self.max_idletime}s"
                            f" for executor {label}; not scaling in")

            # scale by job requests (evolving workflow logic)
            if check_job_request_exists(self.job_requests_file, job_id) or check_elasticity_active(self.policy_file, worker_change_file, job_id):
                logger.info("Job Request already pending or elastic adjustment active.")
            else:
                if isinstance(executor.provider.launcher, PMIxLauncher):
                    active_slots = self.current_tasks_per_node * self.current_nodes_per_block
                else:
                    active_slots = self.current_nodes_per_block

                logger.info(f"Executor has active tasks of {active_tasks} and active slots of {active_slots}")

                parallelism = executor.provider.parallelism
                script_path = executor.provider.script_dir
                worker_change_file = f"{script_path}/worker_change_file"

                if (float(active_slots) / active_tasks) < parallelism:
                    if isinstance(executor.provider.launcher, SimplePMIxLauncher):
                        nodes_to_add = int(active_tasks - active_slots)
                    else:
                        nodes_to_add = int((active_tasks - active_slots) / self.current_tasks_per_node)
                    
                    if nodes_to_add > 0:
                        logger.info("Trying Expansion Request")
                        if self.current_nodes_per_block < executor.provider.max_nodes:
                            if not check_job_request_exists(self.job_requests_file, job_id) and not check_elasticity_active(self.policy_file, worker_change_file, job_id):
                                update_job_requests_file(self.job_requests_file, "expand", nodes_to_add, job_id)
                            else:
                                logger.info("Job Request already pending or elastic adjustment active.")
                        else:
                            logger.info(f"Job {job_id} already at full possible allocation. No Further Expansion.")

                elif (float(active_slots) / active_tasks) > parallelism:
                    if isinstance(executor.provider.launcher, SimplePMIxLauncher):
                        nodes_to_remove = int(active_slots - active_tasks)
                    else:
                        nodes_to_remove = int((active_slots - active_tasks) / self.current_tasks_per_node)

                    if nodes_to_remove > 0:
                        logger.info("Trying Shrinkage Request")
                        if self.current_nodes_per_block > executor.provider.min_nodes:
                            if not check_job_request_exists(self.job_requests_file, job_id) and not check_elasticity_active(self.policy_file, worker_change_file, job_id):
                                update_job_requests_file(self.job_requests_file, "shrink", nodes_to_remove, job_id)
                            else:
                                logger.info("Job Request already pending or elastic adjustment active.")
                        else:
                            logger.info(f"Job {job_id} already at minimum possible allocation. No Further Shrinking.")
                else:
                    pass

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
