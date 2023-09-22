from mpi4py import MPI
import time
import os
import numpy as np
rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD
import pydevd_pycharm
port_mapping=[41423,41125]
pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
from Envs.normalize_wrapper import NormalizeWrapper
import gym

env=gym.make("tactile_push/PushBall-v1",render_mode="human",seed=rank)
wrapperd_env=NormalizeWrapper(env)
observation=wrapperd_env.reset()
print(observation)

# create a shared array of size 1000 elements of type double
size = 100
itemsize = MPI.DOUBLE.Get_size()
if comm.Get_rank() == 0:
    nbytes = size * itemsize
else:
    nbytes = 0

# on rank 0, create the shared block
# on rank 1 get a handle to it (known as a window in MPI speak)
win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

buf, itemsize = win.Shared_query(0)
assert itemsize == MPI.DOUBLE.Get_size()
ary = np.ndarray(buffer=buf, dtype='d', shape=(size,))

if comm.rank == 1:
    ary[:5] = np.arange(5)

# wait in process rank 0 until process 1 has written to the array
comm.Barrier()

# check that the array is actually shared and process 0 can see
# the changes made in the array by process 1
if comm.rank == 0:
    print(ary[:10])