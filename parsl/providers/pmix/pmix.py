import os
import signal
import time
import logging
import copy
import subprocess
import typeguard

from typing import Optional

from parsl.launchers.base import Launcher
from parsl.jobs.states import JobState, JobStatus
from parsl.utils import RepresentationMixin
from parsl.providers.cluster_provider import ClusterProvider

from parsl.launchers import PMIxLauncher

logger = logging.getLogger(__name__)


def write_hostfile(nodes: list, hostfile_path: str, slots: int, scale: Optional[str] = None) -> bool:
    """Write node identifiers to a hostfile for PMIx/OpenMPI.

    Parameters
    ----------
    nodes : list
        List of node identifiers to write to the hostfile
    hostfile_path : str
        Path to the hostfile to be created or overwritten
    slots : int
        Number of slots per node (cores/processes)
    scale : Optional[str]
        Scaling operation type:
        - None: Regular slot assignment (slots=N)
        - "expand": Additive slot assignment (slots=+N)
        - "shrink": Subtractive slot assignment (slots=-N)

    Returns
    -------
    bool
        True if hostfile was successfully written, False otherwise
    """
    if not nodes:
        logger.warning(f"No nodes provided when writing hostfile {hostfile_path}")
        return False
    
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(hostfile_path)), exist_ok=True)
        
        with open(hostfile_path, 'w') as file:
            # Use elif for mutually exclusive conditions
            if scale is None:
                file.writelines(f"{node.strip()} slots={slots}\n" for node in nodes)
            elif scale == "shrink":
                file.writelines(f"{node.strip()} slots=-{slots}\n" for node in nodes)
            elif scale == "expand":
                file.writelines(f"{node.strip()} slots=+{slots}\n" for node in nodes)
            else:
                logger.warning(f"Invalid scale value '{scale}' when writing hostfile {hostfile_path}")
                file.writelines(f"{node.strip()} slots={slots}\n" for node in nodes)
                
        logger.debug(f"Hostfile written to {hostfile_path} with {len(nodes)} nodes, slots={slots}, scale={scale}")
        return True
        
    except IOError as e:
        logger.error(f"Failed to write hostfile {hostfile_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing hostfile {hostfile_path}: {e}")
        return False

def launch(run_command):
    local_env = os.environ.copy()
    envs = copy.deepcopy(local_env)
    proc = subprocess.Popen(
        run_command,
        env=envs,
        close_fds=True,
        shell=True,
        start_new_session=True  # creates a new pgid == proc.pid
    )
    return proc

def verify_process_running(pid, timeout=5):
    """Verify process is still running after brief delay"""
    start_time = time.time()
    time.sleep(0.5)  # Short initial delay
    
    while time.time() - start_time < timeout:
        try:
            # os.kill with signal 0 just checks if process exists
            os.kill(pid, 0)
            return True  # Process is running
        except ProcessLookupError:
            time.sleep(0.5)  # Process might be starting
    
    return False  # Process didn't start or died quickly

def _terminate_pgid(pgid: int, grace_seconds: float = 3.0) -> bool:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True  # already gone
    except Exception as e:
        logger.warning("Failed to send SIGTERM to pgid %s: %s", pgid, e)

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        try:
            # sending signal 0 checks if any process in the group still exists
            os.killpg(pgid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            return True  # group exited

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception as e:
        logger.error("Failed to send SIGKILL to pgid %s: %s", pgid, e)
        return False

    time.sleep(0.1)
    return True


def start_dvm(local_hostfile, dvm_uri):
    # run DVM
    local_env = os.environ.copy()
    envs = copy.deepcopy(local_env)
    cmd = "/home/rbhattara/pmix_recent/install/prrte/bin/prte --pmixmca ptl_base_if_include ib0 --report-uri {0} --hostfile {1} --prtemca plm ^slurm --daemonize".format(
        dvm_uri, local_hostfile)
    logger.info(cmd)
    proc = subprocess.run(
        cmd,
        env=envs,
        capture_output=True,
        shell=True
    )
    logger.info("PRRTE DVM Started")


def stop_dvm(dvm_uri):
    # stop DVM
    local_env = os.environ.copy()
    envs = copy.deepcopy(local_env)
    cmd = "pterm --dvm-uri file:{0}".format(dvm_uri)
    logger.info(cmd)
    proc = subprocess.run(
        cmd,
        env=envs,
        capture_output=True,
        shell=True
    )
    logger.info(f"PRRTE DVM Terminated: {cmd} {proc.stdout} {proc.stderr}")


class PMIxProvider(ClusterProvider, RepresentationMixin):
    """PMIx Execution Provider
    """
    @typeguard.typechecked
    def __init__(self,
                 nodes_per_block: int = 1,
                 cores_per_node: Optional[int] = 64,
                 init_blocks: int = 1,
                 min_blocks: int = 0,
                 max_blocks: int = 1,
                 min_nodes: int = 1,
                 max_nodes: int = 1,
                 parallelism: float = 1,
                 job_id=-1,
                 node_list: str = '',
                 walltime: str = "00:10:00",
                 worker_init_env: str = '',
                 cmd_timeout: int = 10,
                 launcher: Launcher = PMIxLauncher(),):

        label = 'pmix'
        super().__init__(label,
                         nodes_per_block,
                         init_blocks,
                         min_blocks,
                         max_blocks,
                         parallelism,
                         walltime,
                         cmd_timeout=cmd_timeout,
                         launcher=launcher)

        self.job_id = job_id
        self.nodes = init_blocks * nodes_per_block
        self.node_list = node_list.split(",")
        self.cores_per_node = cores_per_node
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.elastic_nodes_id = 0
        self.worker_init_env = worker_init_env

    def _status(self):
        '''Returns the status list for a list of job_ids

        Args:
              self

        Returns:
              [status...] : Status list of all jobs
        '''
        return

    def submit(self, command, tasks_per_node: int = 1, job_name: str = "parsl.pmix"):
        """Submit the command as a pmix job.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        tasks_per_node : int
            Command invocations to be launched per node
        job_name : str
            Name for the job
        Returns
        -------
        None or str
            If at capacity, returns None; otherwise, a string identifier for the job
        """

        script_path = self.script_dir
        script_path = os.path.abspath(script_path)
        # hostfile and dvm file paths
        local_hostfile = "{0}/hostfile".format(script_path)
        dvm_uri = "{0}/dvm.uri".format(script_path)

        write_hostfile(self.node_list, local_hostfile, self.cores_per_node)

        start_dvm(local_hostfile, dvm_uri)

        new_command = self.launcher(
            command, self.nodes_per_block, dvm_uri, local_hostfile, self.worker_init_env)
        logger.info("Command prun %s", new_command)

        proc = launch(new_command)
        time.sleep(1)
        logger.info("Allocated with jobid: %s and pid %s", self.job_id, proc.pid)
        if not verify_process_running(proc.pid, timeout=5):
            logger.error(f"Process {proc.pid} failed to start or died quickly")
            # Handle failure - maybe retry or raise exception
            raise RuntimeError(f"Failed to start process for job {self.job_id}")

        self.resources[self.job_id] = {'job_id': self.job_id, 'status': JobStatus(
            JobState.RUNNING), 'pid_and_nodes': [(proc.pid, self.node_list)]}

        return self.job_id

    def submit_resource_change(self, command, scale, num_nodes, nodes, job_id, elasticity_type="manager"):
        """Submit the command as a pmix job change.

        Parameters
        ----------
        command : str
            Command to be made on the remote side.
        Returns
        -------
        None
        """

        script_path = self.script_dir
        script_path = os.path.abspath(script_path)
        # hostfile and dvm file paths
        dvm_uri = "{0}/dvm.uri".format(script_path)
        logger.info("Elasticity Type for resource change is %s", elasticity_type)

        if elasticity_type == "manager" and scale == "expand":
            for node in nodes:
                local_add_hostfile = "{0}/add_hostfile{1}".format(
                    script_path, self.elastic_nodes_id)
                self.elastic_nodes_id += 1

                write_hostfile([node], local_add_hostfile, self.cores_per_node, scale="expand")
                run_command = (
                    f"prun -x DVM_URI={dvm_uri} --dvm-uri file:{dvm_uri} "
                    f"--add-hostfile {local_add_hostfile} "
                    f"--host {node}:{self.cores_per_node} "
                    f"--map-by node --bind-to none -n 1 " 
                    f"{self.worker_init_env}/bin/python {self.worker_init_env}/bin/{command} &"
                )
                proc = launch(run_command)
                logger.info("Launched expansion command: %s", run_command)
                # fix parallel runs bug on dvm change
                time.sleep(1)
                if not verify_process_running(proc.pid, timeout=5):
                    logger.error(f"Process {proc.pid} failed to start or died quickly")
                    raise RuntimeError(f"Failed to start process for job {self.job_id}")
                logger.info(f"Process {proc.pid} started successfully for node {node}")

                logger.info("Allocated with node: %s on job id: %s", node, job_id)
                logger.info(f"Finished expansion of jobid: {job_id} with node {node}")
                self.resources[job_id]['pid_and_nodes'].append((proc.pid, [node.strip()]))

        if elasticity_type == "manager" and scale == "shrink":
            node_to_kill_file_path = f"{script_path}/node_to_kill_file"
            local_add_hostfile = f"{script_path}/add_hostfile{self.elastic_nodes_id}"
            self.elastic_nodes_id += 1

            write_hostfile(nodes, local_add_hostfile, self.cores_per_node, scale="shrink")
            pid_and_nodes = self.resources[job_id]['pid_and_nodes']

            # Process each node to shrink
            for node in nodes:
                node_stripped = node.strip()
                matching_entries = [(pid, pid_nodes) for pid, pid_nodes in pid_and_nodes 
                                  if node_stripped in pid_nodes]
                
                if not matching_entries:
                    logger.warning("Node %s not found in job %s resources", node_stripped, job_id)
                    continue

                # Write node to kill file (hint for worker signal handler)
                try:
                    with open(node_to_kill_file_path, 'w') as file:
                        file.write(f"{node_stripped}\n")
                except Exception as e:
                    logger.debug("Failed to write node_to_kill_file for %s: %s", node_stripped, e)

                # Terminate processes on this node
                for pid, pid_nodes in matching_entries:
                    logger.info("Found node %s with pid %s for shrink", node_stripped, pid)
                    
                    # Send SIGURG hint to worker (gentle shutdown)
                    actual_pid = pid + 1
                    try:
                        os.kill(actual_pid, signal.SIGURG)
                        logger.info("Sent SIGURG to pid %d for node %s", actual_pid, node_stripped)
                    except ProcessLookupError:
                        logger.info("Process %s already exited", actual_pid)
                        continue
                    except Exception as e:
                        logger.debug("SIGURG to %d failed: %s", actual_pid, e)

                    # Give worker a moment to handle the signal
                    time.sleep(0.5)

                    # Terminate the process group robustly
                    try:
                        pgid = os.getpgid(pid)
                        if not _terminate_pgid(pgid):
                            logger.warning("Failed to terminate pgid %s for pid %s", pgid, pid)
                    except ProcessLookupError:
                        logger.info("Process %s already exited", pid)
                    except Exception as e:
                        logger.warning("Termination failed for pid %s: %s", pid, e)

                # Remove entries for this node from tracking
                self.resources[job_id]['pid_and_nodes'] = [
                    (pid, pid_nodes) for pid, pid_nodes in pid_and_nodes 
                    if node_stripped not in pid_nodes
                ]
                pid_and_nodes = self.resources[job_id]['pid_and_nodes']  # Update local reference

                logger.info("Finished shrinkage of jobid: %s with node %s", job_id, node_stripped)

            # Optional: Update DVM hostfile to reflect the shrinkage
            while os.path.getsize(node_to_kill_file_path) > 0:
                time.sleep(1)
            try:
                run_command = f"prun --dvm-uri file:{dvm_uri} --add-hostfile {local_add_hostfile} -n 1 hostname &"
                proc = launch(run_command)
                time.sleep(1)
            except Exception as e:
                logger.warning("DVM hostfile update failed: %s", e)
            logger.info("Updating DVM hostfile to reflect shrinkage")
            time.sleep(2)  # Give DVM a moment to process the change

        if elasticity_type == "worker":
            worker_change_file = f"{script_path}/worker_change_file"
            local_add_hostfile = "{0}/add_hostfile{1}".format( script_path, self.elastic_nodes_id)
            self.elastic_nodes_id += 1
            if scale == "expand":
                write_hostfile(nodes, local_add_hostfile, self.cores_per_node, scale="expand")
            elif scale == "shrink":
                write_hostfile(nodes, local_add_hostfile, self.cores_per_node, scale="shrink")
            else:
                logger.info("Incorrect scaling type.")
            logger.info("Scaling the DVM with add hosts")
            with open(worker_change_file, 'w') as file:
                file.write(f"{num_nodes} {scale} {local_add_hostfile}\n")
            while os.path.getsize(worker_change_file) > 0:
                time.sleep(1)
            logger.info(f"Finished {scale} of jobid: {job_id} with nodes {nodes}")

    def cancel(self, job_ids):
        ''' Cancels the jobs specified by a list of job ids

        Args:
        job_ids : [<job_id> ...]

        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        '''

        script_path = os.path.abspath(self.script_dir)
        dvm_uri = f"{script_path}/dvm.uri"

        results = []

        for jid in job_ids:
            ok = True
            res = self.resources.get(jid)
            if not res:
                logger.warning("Cancel called for unknown job id: %s", jid)
                results.append(False)
                continue

            pid_and_nodes = res.get('pid_and_nodes', [])
            for pid in pid_and_nodes:
                try:
                    pgid = os.getpgid(pid)
                except ProcessLookupError:
                    logger.info("Process %s already exited (jid %s)", pid, jid)
                    continue
                except Exception as e:
                    logger.warning("Could not get pgid for pid %s (jid %s): %s", pid, jid, e)
                    # Best-effort: try both TERM and KILL on the pid
                    try:
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(0.2)
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception as ee:
                        logger.error("Killing pid %s failed: %s", pid, ee)
                        ok = False
                    continue

                if not _terminate_pgid(pgid):
                    logger.error("Failed to terminate process group %s for jid %s", pgid, jid)
                    ok = False

            if ok:
                self.resources[jid]['status'] = JobStatus(JobState.CANCELLED)
                logger.info("Killed Job: %s", jid)
            else:
                logger.warning("Job %s may not have been fully terminated", jid)

            results.append(ok)

        try:
            stop_dvm(dvm_uri)
        except Exception as e:
            logger.warning("Failed to stop DVM: %s", e)

        return results

    @property
    def status_polling_interval(self):
        return 30
