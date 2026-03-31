// Package kernel of GoSeek-V3 - A Go runtime for DeepSeek models.
//
// This file is a Go rewrite of the original Python implementation
// designed to run DeepSeek models.
// Original by DeepSeek: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py
//
// Modified by Lennard <leycm@proton.me>, 2026.
//
// This project is not affiliated with or endorsed by DeepSeek.
package kernel

import "fmt"

// BlockSizeError is returned by ActQuant when the total number of elements
// is not divisible by the specified block size.
// Use errors.As to extract Numel and BlockSize.
type BlockSizeError struct {
	Numel     int64 // Total number of elements in the tensor
	BlockSize int   // Block size that caused the error
}

func (e BlockSizeError) Error() string {
	return fmt.Sprintf("kernel: numel %d is not divisible by blockSize %d", e.Numel, e.BlockSize)
}

// NewBlockSizeError constructs a new BlockSizeError.
func NewBlockSizeError(numel int64, blockSize int) error {
	return BlockSizeError{Numel: numel, BlockSize: blockSize}
}

// WeightDivisibilityError occurs when the dimensions of a weight matrix
// (M, N) are not divisible by the block size used for dequantization.
type WeightDivisibilityError struct {
	M, N      int64 // Dimensions of the weight matrix
	BlockSize int   // Block size causing the error
}

func (e WeightDivisibilityError) Error() string {
	return fmt.Sprintf("kernel: M=%d and N=%d must both be divisible by blockSize=%d", e.M, e.N, e.BlockSize)
}

// NewWeightDivisibilityError constructs a new WeightDivisibilityError.
func NewWeightDivisibilityError(M, N int64, blockSize int) error {
	return WeightDivisibilityError{M: M, N: N, BlockSize: blockSize}
}

// WeightScaleMismatchError occurs when the number of scale factors
// does not match the expected number based on the tensor shape.
type WeightScaleMismatchError struct {
	Expected int // Expected number of scale values
	Got      int // Actual number of scale values provided
}

func (e WeightScaleMismatchError) Error() string {
	return fmt.Sprintf("kernel: expected %d scale values, got %d", e.Expected, e.Got)
}

// NewWeightScaleMismatchError constructs a new WeightScaleMismatchError.
func NewWeightScaleMismatchError(expected, got int) error {
	return WeightScaleMismatchError{Expected: expected, Got: got}
}

// FP8GEMMInvalidKError occurs when the K dimension of the FP8 GEMM
// input is not divisible by the block size.
type FP8GEMMInvalidKError struct {
	K, BlockSize int64 // K dimension and block size causing the error
}

func (e FP8GEMMInvalidKError) Error() string {
	return fmt.Sprintf("kernel: K=%d must be divisible by blockSize=%d", e.K, e.BlockSize)
}

// NewFP8GEMMInvalidKError constructs a new FP8GEMMInvalidKError.
func NewFP8GEMMInvalidKError(K, blockSize int64) error {
	return FP8GEMMInvalidKError{K: K, BlockSize: blockSize}
}

// FP8GEMMInvalidNError occurs when the N dimension of the FP8 GEMM
// input is not divisible by the block size.
type FP8GEMMInvalidNError struct {
	N, BlockSize int64 // N dimension and block size causing the error
}

func (e FP8GEMMInvalidNError) Error() string {
	return fmt.Sprintf("kernel: N=%d must be divisible by blockSize=%d", e.N, e.BlockSize)
}

// NewFP8GEMMInvalidNError constructs a new FP8GEMMInvalidNError.
func NewFP8GEMMInvalidNError(N, blockSize int64) error {
	return FP8GEMMInvalidNError{N: N, BlockSize: blockSize}
}
