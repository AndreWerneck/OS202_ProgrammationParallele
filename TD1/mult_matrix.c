#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

struct timeval tv;

void mult_tp(int m3[], int m1[], int m2[], int n) {
    int *m2t = malloc(n*n * sizeof(int));
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            m2t[j*n + i] = m2[i*n + j];
        }
    }
    for (int i = 0;i < n;i++) {
        for (int j = 0;j < n;j++) {
            for (int k = 0;k < n;k++) {
                m3[i*n + j] += m1[i*n + k] * m2t[j*n + k]; 
            }
        }
    }

    free(m2t);
}

// m3 = m1 * m2;
void mult_ijk(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int i = 0;i < n;i++, misses+=2) {
        for (int j = 0;j < n;j++) {
            for (int k = 0;k < n;k++, misses++) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j];
            }
        }
    }
    printf("Misses: %ld\n", misses);
}

void mult_ikj(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int i = 0;i < n;i++, misses+=2) {
        for (int k = 0;k < n;k++, misses++) {
            for (int j = 0;j < n;j++) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j]; 
            }
        }
    }
    printf("Misses: %ld\n", misses);
}

void mult_jik(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int j = 0;j < n;j++) {
        for (int i = 0;i < n;i++, misses+=2) {
            for (int k = 0;k < n;k++, misses++) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j]; 
            }
        }
    }
    printf("Misses: %ld\n", misses);
}

void mult_jki(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int j = 0;j < n;j++) {
        for (int k = 0;k < n;k++, misses++) {
            for (int i = 0;i < n;i++, misses+=2) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j]; 
            }
        }
    }
    printf("Misses: %ld\n", misses);
}

void mult_kij(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int k = 0;k < n;k++, misses++) {
        for (int i = 0;i < n;i++, misses += 2) {
            for (int j = 0;j < n;j++) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j]; 
            }
        }
    }
    printf("Misses: %ld\n", misses);
}

void mult_kji(int m3[], int m1[], int m2[], int n) {
    long misses = 0;
    for (int k = 0;k < n;k++, misses++) {
        for (int j = 0;j < n;j++) {
            for (int i = 0;i < n;i++, misses += 2) {
                m3[i*n + j] += m1[i*n + k] * m2[k*n + j]; 
            }
        }
    }
    printf("Misses: %ld\n", misses);
} 

void clear(int m[], int n) {
    for (int i = 0;i < n*n;i++) m[i] = 0.0;
}

void print(int m[], int n) {
    for (int i = 0;i < n*n;i++) printf("%d,", m[i]);
    printf("\n");
}

void start_timer() {
    gettimeofday(&tv,NULL);
}

// returns time diff in seconds as a float
float stop_timer() {
    long before = tv.tv_sec * 1000 + tv.tv_usec / 1000; // milliseconds
    gettimeofday(&tv,NULL);
    long after = tv.tv_sec * 1000 + tv.tv_usec / 1000; // milliseconds
    float diff = (after - before) / 1000.0;
    return diff;
}

int main(int argc, char *argv []) {
    int n = atoi(argv[1]);
    
    // for (int i = 0; i <= 1024 + 512 + 1; i++) {
    printf("N: %d\n", n);
    int *m1 = malloc(n*n * sizeof(int));
    int *m2 = malloc(n*n * sizeof(int));
    int *m3 = malloc(n*n * sizeof(int));
    float diff = 0.0;
    int t = 1;

    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_ijk(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("IJK ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_ikj(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("IKJ ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_jik(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("JIK ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_jki(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("JKI ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_kij(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("KIJ ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_kji(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("KJI ; Diff: %f\n\n", diff / t);

    diff = 0.0;
    for (int i = 0;i < t;i++) {
        clear(m3, n);
        start_timer();
        mult_tp(m3, m1, m2, n);
        diff += stop_timer();
    }
    printf("TP ; Diff: %f\n\n", diff / t);

    free(m1);
    free(m2);
    free(m3);

    //     n += 1;
    // }
    
    return 0;
}

8.6*10^9
4.2*10^6
8.6*10^9
17.2*10^9
8.4*10^6
17.2*10^9