#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {

void progSubBlocksParallel(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C, int num_threads) {
  int i, j, k;
  omp_set_num_threads(num_threads);
  #pragma omp parallel
  {
    #pragma omp for
    for (i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
      for (j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
        for (k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
          C(i, j) += A(i, k) * B(k, j);
  }
}

void progParallelOptimized(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  int i, j, k;
  omp_set_num_threads(8);
  #pragma omp parallel
  {
    #pragma omp for
    for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
      for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
        for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
          C(i, j) += A(i, k) * B(k, j);
  }
}

void prodSubBlocksIJK(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
    for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
      for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocksIKJ(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
      for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocksJIK(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
    for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
      for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocksJKI(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
      for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocksKIJ(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
    for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
      for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
        C(i, j) += A(i, k) * B(k, j);
}
void prodSubBlocksKJI(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
    for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
      for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
        C(i, j) += A(i, k) * B(k, j);
}
void prodBlocksParallel(int szBlock, const Matrix& A, const Matrix& B, Matrix& C) {
  int iRowBlkA, iColBlkB, iColBlkA;
  #pragma omp parallel
  {
    #pragma omp for
    for (iRowBlkA = 0; iRowBlkA < A.nbRows; iRowBlkA += szBlock)
      for (iColBlkB = 0; iColBlkB < B.nbCols; iColBlkB += szBlock)
        for (iColBlkA = 0; iColBlkA < A.nbCols; iColBlkA += szBlock)
          prodSubBlocksJKI(iRowBlkA, iColBlkB, iColBlkA, szBlock, A, B, C);
  }
}
void prodBlocks(int szBlock, const Matrix& A, const Matrix& B, Matrix& C) {
  int iRowBlkA, iColBlkB, iColBlkA;
  for (iRowBlkA = 0; iRowBlkA < A.nbRows; iRowBlkA += szBlock)
    for (iColBlkB = 0; iColBlkB < B.nbCols; iColBlkB += szBlock)
      for (iColBlkA = 0; iColBlkA < A.nbCols; iColBlkA += szBlock)
        prodSubBlocksJKI(iRowBlkA, iColBlkB, iColBlkA, szBlock, A, B, C);
}
const int szBlock = 32;
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  int szBlock = 256;
  // prodBlocks(szBlock, A, B, C);
  // prodSubBlocksJKI(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  // progSubBlocksParallel(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C, 16);
  progParallelOptimized(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  return C;
}
