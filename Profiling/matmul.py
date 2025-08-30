#!/usr/bin/env python3
"""
Matrix multiplication using pure Python implementation.
"""

import random


def generate_matrix(size):
    """
    Generates a square matrix of a given size with random integer values.
    
    Args:
        size (int): The size of the square matrix to generate
        
    Returns:
        list: A 2D list representing a square matrix with random values 0-100
    """
    matrix = []
    for _ in range(size):
        row = [random.randint(0, 100) for _ in range(size)]
        matrix.append(row)
    return matrix


def multiply_matrices_pure_python(A, B):
    """
    Multiplies two matrices using pure Python nested loops.
    
    Args:
        A (list): First matrix (2D list)
        B (list): Second matrix (2D list)
        
    Returns:
        list: Result matrix C where C = A * B
    """
    # Get the size of the matrices (assuming they are square and of the same size)
    size = len(A)
    
    # Initialize the result matrix with zeros
    C = [[0 for _ in range(size)] for _ in range(size)]
    
    # Perform matrix multiplication
    # Iterate through rows of A
    for i in range(size):
        # Iterate through columns of B
        for j in range(size):
            # Iterate through rows of B
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]
                
    return C


def main():
    """
    Main function to demonstrate matrix multiplication.
    """
    MATRIX_SIZE = 1000  # Reduced size for faster execution
    
    print(f"Generating two {MATRIX_SIZE}x{MATRIX_SIZE} matrices...")
    matrix_a = generate_matrix(MATRIX_SIZE)
    matrix_b = generate_matrix(MATRIX_SIZE)
    
    # Print a small sample of each matrix for verification
    print("Sample from matrix A (top-left 3x3):")
    for i in range(min(3, len(matrix_a))):
        row_sample = [matrix_a[i][j] for j in range(min(3, len(matrix_a[0])))]
        print(row_sample)
    
    print("Sample from matrix B (top-left 3x3):")
    for i in range(min(3, len(matrix_b))):
        row_sample = [matrix_b[i][j] for j in range(min(3, len(matrix_b[0])))]
        print(row_sample)
    
    print("Starting matrix multiplication...")
    result_matrix = multiply_matrices_pure_python(matrix_a, matrix_b)
    
    print("Matrix multiplication completed!")
    print(f"Result matrix size: {len(result_matrix)}x{len(result_matrix[0])}")
    
    # Print a small sample of the result for verification
    print("Sample from result matrix (top-left 3x3):")
    for i in range(min(3, len(result_matrix))):
        row_sample = [result_matrix[i][j] for j in range(min(3, len(result_matrix[0])))]
        print(row_sample)


if __name__ == "__main__":
    main()