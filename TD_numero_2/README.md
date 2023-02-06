# TP2 de NOM Prénom

`pandoc -s --toc tp2.md --css=./github-pandoc.css -o tp2.html`





## Mandelbrot 

*Expliquer votre stratégie pour faire une partition équitable des lignes de l'image entre chaque processus*

           | Taille image : 800 x 600 | 
-----------+---------------------------
séquentiel |              
1          |              
2          |              
3          |              
4          |              
8          |              


*Discuter sur ce qu'on observe, la logique qui s'y cache.*

*Expliquer votre stratégie pour faire une partition dynamique des lignes de l'image entre chaque processus*

           | Taille image : 800 x 600 | 
-----------+---------------------------
séquentiel |              
1          |              
2          |              
3          |              
4          |              
8          |              



## Produit matrice-vecteur



*Expliquer la façon dont vous avez calculé la dimension locale sur chaque processus, en particulier quand le nombre de processus ne divise pas la dimension de la matrice.*

Le code suivant a été utilisé pour calculer la dimension de travail de chaque processus Nloc.
Le travail total à faire est divisé par la quantité de processus MPI. 
Et le dernier processus est responsable pour le reste, quand le nombre de processus ne divise pas la dimension de la matrice.

Example:  

Total lines: 32  
Total process: 3  
Rank work size: 32 / 3 = 10  
  
Rank 0:  
    Begin: 0, End: 10  
Rank 1:  
    Begin: 10, End: 20  
Rank 2:  
    Begin: 20, End 32  

```c
int total_work_size = mul_type == row ? m_nrows : m_ncols;
int rank_work_size = (total_work_size / n_ranks);
int begin = m_rank * rank_work_size;
int end = (m_rank == n_ranks - 1 ? total_work_size : begin + rank_work_size);
```

La résolution des autres questions se trouvent dans le fichier TD2_OS202.pdf