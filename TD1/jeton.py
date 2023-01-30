from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank
name = MPI.Get_processor_name()
print(f"Je suis le processus {rank} sur {nbp} processus")
print(f"Je m'execute sur l'ordinateur {name}")
liste_recue = None

if rank == 0:
    jeton = rank + 1
    req = globCom.isend(jeton, dest=rank+1)
    liste_recue = globCom.recv(source=nbp-1)
    req.wait()
    print(f'Last Received = {liste_recue}')
elif rank == (nbp-1):
    jeton = rank + 1
    req = globCom.isend(jeton, dest=0)
    liste_recue = globCom.recv(source=rank - 1)
else:
    jeton = rank + 1
    req = globCom.isend(jeton, dest=rank+1)
    liste_recue = globCom.recv(source=rank-1)
    req.wait()