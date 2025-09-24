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


def write_hostfile(nodes, hostfile_path, slots, shrink=False):
    """Write node identifiers back to the hostfile."""
    with open(hostfile_path, 'w') as file:
        if shrink:
            file.writelines(
                f"{node.strip()} slots=-{slots} \n" for node in nodes)
        else:
            file.writelines(
                f"{node.strip()} slots={slots} \n" for node in nodes)

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
    cmd = "prte --pmixmca ptl_base_if_include ib0 --report-uri {0} --hostfile {1} --prtemca plm ^slurm --daemonize".format(
        dvm_uri, local_hostfile)  # --pmixmca ptl_base_if_include ib0
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
                 cores_per_node: Optional[int] = None,
                 init_blocks: int = 1,
                 min_blocks: int = 0,
                 max_blocks: int = 1,
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

        write_hostfile(self.node_list, local_hostfile, self.cores_per_node, False)

        start_dvm(local_hostfile, dvm_uri)

        new_command = self.launcher(
            command, self.nodes_per_block, dvm_uri, local_hostfile, self.worker_init_env)
        logger.info("Command prun %s", new_command)

        proc = launch(new_command)
        logger.info("Allocated with jobid: %s", self.job_id)

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

        if elasticity_type == "manager":
            for node in nodes:
                local_add_hostfile = "{0}/add_hostfile{1}".format(
                    script_path, self.elastic_nodes_id)
                self.elastic_nodes_id += 1

                if scale == "expand":
                    write_hostfile([node], local_add_hostfile, self.cores_per_node)
                    run_command = (
                        f"prun --dvm-uri file:{dvm_uri} "
                        f"--add-hostfile {local_add_hostfile} "
                        f"--hostfile {local_add_hostfile} "
                        f"--map-by node --bind-to none -n 1 "
                        f"{self.worker_init_env}/bin/python {self.worker_init_env}/bin/{command} &"
                    )
                    proc = launch(run_command)
                    # fix parallel runs bug on dvm change
                    time.sleep(1)
                    logger.info("Allocated with node: %s on job id: %s", node, job_id)
                    print(f"Finished expansion of jobid: {job_id} with node {node}")
                    self.resources[job_id]['pid_and_nodes'].append((proc.pid, [node.strip()]))
                elif scale == "shrink":
                    node_to_kill_file_path = f"{script_path}/node_to_kill_file"
                    write_hostfile([node], local_add_hostfile, self.cores_per_node, shrink=True)

                    pid_and_nodes = self.resources[job_id]['pid_and_nodes']

                    for pid, pid_nodes in pid_and_nodes:
                        if node.strip() in pid_nodes:
                            logger.info("Found node and pid {node} {pid}")
                            with open(node_to_kill_file_path, 'w') as file:
                                file.write(f"{node}\n")
                            actual_pid = int(pid) + 1
                            # Terminate the process
                            try:
                                # Try to terminate the process gracefully
                                os.kill(actual_pid, signal.SIGURG)
                                logger.info("Killing Gracefully %d Check.", pid)
                            except ProcessLookupError:
                                print(f"No process with PID {pid} found.")
                            run_command = f"prun --dvm-uri file:{dvm_uri} --add-hostfile {local_add_hostfile} -n {1} hostname &"
                            proc = launch(run_command)

                            self.resources[job_id]['pid_and_nodes'] = [
                                tup for tup in pid_and_nodes if node not in tup[1]]
                            print(
                                f"Finished shrinkage of jobid: {job_id} with node {node}")
                            # fix parallel runs bug on dvm change
                            time.sleep(1)
                else:
                    logger.info("Incorrect scaling instruction.")
        elif elasticity_type == "worker":
            worker_change_file = f"{script_path}/worker_change_file"
            local_add_hostfile = "{0}/add_hostfile{1}".format( script_path, self.elastic_nodes_id)
            self.elastic_nodes_id += 1
            if scale == "expand":
                write_hostfile(nodes, local_add_hostfile, self.cores_per_node)
            elif scale == "shrink":
                write_hostfile(nodes, local_add_hostfile, self.cores_per_node, shrink=True)
            else:
                logger.info("Incorrect scaling type.")
            logger.info("Scaling the DVM with add hosts")
            with open(worker_change_file, 'w') as file:
                file.write(f"{num_nodes} {scale} {local_add_hostfile}\n")
        else:
            logger.info("Incorrect elasticity type.")

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
