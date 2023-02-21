// Produit matrice-vecteur
# include <cassert>
# include <vector>
# include <iostream>
# include <mpi.h>

using namespace std;

// MPI Unique rank is assigned to each process in a communicator
int m_rank;

// MPI Total numbers of ranks
int n_ranks;

// Define multiplication type: by row or by col
enum MulType { col, row };
MulType mul_type = row;

// ---------------------------------------------------------------------
class Matrix : public std::vector<double>
{
public:
    Matrix (int dim);
    Matrix( int nrows, int ncols );
    Matrix( const Matrix& A ) = delete;
    Matrix( Matrix&& A ) = default;
    ~Matrix() = default;

    Matrix& operator = ( const Matrix& A ) = delete;
    Matrix& operator = ( Matrix&& A ) = default;
    
    double& operator () ( int i, int j ) {
        return m_arr_coefs[i + j*m_nrows];
    }
    double  operator () ( int i, int j ) const {
        return m_arr_coefs[i + j*m_nrows];
    }
    
    std::vector<double> operator * ( const std::vector<double>& u ) const;
    
    std::ostream& print( std::ostream& out ) const
    {
        const Matrix& A = *this;
        out << "[\n";
        for ( int i = 0; i < m_nrows; ++i ) {
            out << " [ ";
            for ( int j = 0; j < m_ncols; ++j ) {
                out << A(i,j) << " ";
            }
            out << " ]\n";
        }
        out << "]";
        return out;
    }
private:
    int m_nrows, m_ncols;
    std::vector<double> m_arr_coefs;
};
// ---------------------------------------------------------------------
inline std::ostream& 
operator << ( std::ostream& out, const Matrix& A )
{
    return A.print(out);
}
// ---------------------------------------------------------------------
inline std::ostream&
operator << ( std::ostream& out, const std::vector<double>& u )
{
    out << "[ ";
    for ( const auto& x : u )
        out << x << " ";
    out << " ]";
    return out;
}
// ---------------------------------------------------------------------
std::vector<double> 
Matrix::operator * ( const std::vector<double>& u ) const
{
    const Matrix& A = *this;
    assert( u.size() == unsigned(m_ncols) );
    std::vector<double> v(m_nrows, 0.);

    // Dividing the work equally to all processes
    int total_work_size = mul_type == row ? m_nrows : m_ncols;
    int rank_work_size = (total_work_size / n_ranks);
    int begin = m_rank * rank_work_size;
    int end = (m_rank == n_ranks - 1 ? total_work_size : begin + rank_work_size);

    // if we are doing by lines...
    if (mul_type == row) { // Question 2.4.2 Produit parallèle matrice – vecteur par ligne
        for ( int i = begin; i < end; ++i ) {
            for ( int j = 0; j < m_ncols; ++j ) {
                v[i] += A(i,j)*u[j];
            }            
        }
    } else { // Question 2.4.1 Produit parallèle matrice – vecteur par colonne
        for ( int i = 0; i < m_nrows; ++i ) {
            for ( int j = begin; j < end; ++j ) {
                v[i] += A(i,j)*u[j];
            }            
        }
    }
    

    return v;
}

// =====================================================================
Matrix::Matrix (int dim) : m_nrows(dim), m_ncols(dim),
                           m_arr_coefs(dim*dim)
{
    for ( int i = 0; i < dim; ++ i ) {
        for ( int j = 0; j < dim; ++j ) {
            (*this)(i,j) = (i+j)%dim;
        }
    }
}
// ---------------------------------------------------------------------
Matrix::Matrix( int nrows, int ncols ) : m_nrows(nrows), m_ncols(ncols),
                                         m_arr_coefs(nrows*ncols)
{
    int dim = (nrows > ncols ? nrows : ncols );
    for ( int i = 0; i < nrows; ++ i ) {
        for ( int j = 0; j < ncols; ++j ) {
            (*this)(i,j) = (i+j)%dim;
        }
    }    
}
// =====================================================================
int main( int nargs, char* argv[] )
{   
    // Initializes the MPI execution environment
    MPI_Init(&nargs, &argv);

    // Get this process' rank (process within a communicator)
    // MPI_COMM_WORLD is the default communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    // Get the total number of ranks in the communicator
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    const int N = 120;
    Matrix A(N);
    std::vector<double> u( N );
    for ( int i = 0; i < N; ++i ) u[i] = i+1;
    std::vector<double> result_rank = A*u;
    std::vector<double> result_total(N, 0.);

    // Performing the reduce operation
    MPI_Reduce(&result_rank[0], &result_total[0], N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (m_rank == 0) {
        std::cout  << "A : " << A << std::endl;
        std::cout << " u : " << u << std::endl;
        std::cout << "A.u = " << result_total << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    
    return EXIT_SUCCESS;
}
