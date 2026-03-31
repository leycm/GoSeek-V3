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

import (
	"fmt"
	"math"

	torch "github.com/wangkuiyi/gotorch"
)

// defaultBlockSize is the default block size for quantization and dequantization.
const defaultBlockSize = 128

// fp8Max is the maximum representable value in float8_e4m3fn (448.0).
const fp8Max = 448.0

// minScale prevents division by zero / degenerate scales.
const minScale = 1e-4

// ScaleFmt selects the format used for the block scale factor.
type ScaleFmt int

const (
	// ScaleFmtFloat32 keeps the scale as a plain float32 (default).
	ScaleFmtFloat32 ScaleFmt = iota
	// ScaleFmtUE8M0 rounds the scale up to the nearest power-of-two (UE8M0).
	ScaleFmtUE8M0
)

// ActQuantResult holds the quantized tensor and its per-block scaling factors.
type ActQuantResult struct {
	// Quantized holds the FP8-quantized activations, stored as float32 clamped
	// to [-fp8Max, fp8Max] (Go has no native fp8 dtype).
	Quantized torch.Tensor
	// Scales holds one float32 scale value per block.
	Scales []float32
	// ScaleShape mirrors Python's s.shape = (*x.shape[:-1], x.shape[-1]//blockSize).
	ScaleShape []int64
}

// ActQuant quantises x using block-wise scaling, matching kernel.py act_quant.
//
// Each block of blockSize float32 values along the last dimension is scaled
// independently so the maximum absolute value maps to fp8Max (448).
// scaleFmt controls whether the scale is stored as raw float32 or rounded up
// to the nearest power-of-two (ScaleFmtUE8M0, i.e. "ue8m0").
//
// NOTE: gotorch operates on flat tensors; the caller is responsible for
// tracking the logical shape of the returned Quantised tensor.
func ActQuant(x torch.Tensor, blockSize int, scaleFmt ScaleFmt) (ActQuantResult, error) {
	if blockSize <= 0 {
		blockSize = defaultBlockSize
	}

	// mirrors: x.Numel()
	numel := int64(1)
	for _, dim := range x.Shape() {
		numel *= dim
	}

	if numel%int64(blockSize) != 0 {
		return ActQuantResult{}, NewBlockSizeError(numel, blockSize)
	}

	numBlocks := int(numel) / blockSize
	xData := tensorToFloat32Slice(x, int(numel))
	qData := make([]float32, int(numel))
	scales := make([]float32, numBlocks)

	for b := 0; b < numBlocks; b++ {
		start := b * blockSize
		end := start + blockSize

		// amax = max(|x[start:end]|), clamped to minScale.
		// mirrors: amax = tl.max(tl.abs(x)); amax = tl.maximum(amax, 1e-4)
		amax := float32(minScale)
		for i := start; i < end; i++ {
			v := xData[i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}

		s := amax / fp8Max

		if scaleFmt == ScaleFmtUE8M0 {
			// s = 2^ceil(log2(s))
			// mirrors: exp = tl.math.ceil(tl.math.log2(s)); s = tl.math.exp2(exp)
			exp := math.Ceil(math.Log2(float64(s)))
			s = float32(math.Pow(2, exp))
		}

		scales[b] = s

		// quantize and clamp to [-fp8Max, fp8Max].
		// mirrors: y = x / s; y = y.to(y_ptr.dtype.element_ty)
		for i := start; i < end; i++ {
			v := xData[i] / s
			if v > fp8Max {
				v = fp8Max
			} else if v < -fp8Max {
				v = -fp8Max
			}
			qData[i] = v
		}
	}

	quantized := float32SliceToTensor(qData)

	return ActQuantResult{
		Quantized:  quantized,
		Scales:     scales,
		ScaleShape: []int64{int64(numBlocks)},
	}, nil
}

// WeightDequant dequantises a 2-D weight matrix x using per-block scales s,
// matching kernel.py weight_dequant.
//
// x has logical shape (M, N); s has shape (M/blockSize, N/blockSize).
// Each (blockSize × blockSize) tile of x is multiplied by its scalar in s.
//
// Returns a new float32 tensor containing the dequantised values (flat, shape M*N).
func WeightDequant(x torch.Tensor, s []float32, M, N int64, blockSize int) (torch.Tensor, error) {
	if blockSize <= 0 {
		blockSize = defaultBlockSize
	}

	bs := int64(blockSize)
	if M%bs != 0 || N%bs != 0 {
		return torch.Tensor{}, NewWeightDivisibilityError(M, N, blockSize)
	}

	tilesM := M / bs
	tilesN := N / bs
	expectedScales := int(tilesM * tilesN)
	if len(s) != expectedScales {
		return torch.Tensor{}, NewWeightScaleMismatchError(expectedScales, len(s))
	}

	numel := int(M * N)
	xData := tensorToFloat32Slice(x, numel)
	yData := make([]float32, numel)

	// mirrors: the Triton kernel's 2-D pid_m / pid_n tile loop:
	//   x = tl.load(x_ptr + offs); s = tl.load(s_ptr + pid_m*n + pid_n); y = x * s
	for pm := int64(0); pm < tilesM; pm++ {
		for pn := int64(0); pn < tilesN; pn++ {
			scale := s[pm*tilesN+pn]
			for bm := int64(0); bm < bs; bm++ {
				row := pm*bs + bm
				for bn := int64(0); bn < bs; bn++ {
					col := pn*bs + bn
					idx := row*N + col
					yData[idx] = xData[idx] * scale
				}
			}
		}
	}

	return float32SliceToTensor(yData), nil
}

// FP8GEMMConfig holds the matrix dimensions and block size for FP8GEMM.
type FP8GEMMConfig struct {
	M, N, K   int64
	BlockSize int
}

// FP8GEMM performs a matrix multiplication in (emulated) FP8 precision,
// matching kernel.py fp8_gemm.
//
// a has logical shape (M, K); b has shape (N, K) (row-major, already transposed).
// aS has shape (M, K/blockSize); bS has shape (N/blockSize, K/blockSize).
//
// Result c has flat shape M*N (logical shape M×N).
//
// Mirrors the Triton kernel's accumulator loop:
//
//	accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
func FP8GEMM(
	a torch.Tensor, aS []float32,
	b torch.Tensor, bS []float32,
	cfg FP8GEMMConfig,
) (torch.Tensor, error) {
	bs := int64(cfg.BlockSize)
	if bs == 0 {
		bs = defaultBlockSize
	}

	M, N, K := cfg.M, cfg.N, cfg.K

	if K%bs != 0 {
		return torch.Tensor{}, NewFP8GEMMInvalidKError(K, bs)
	}
	if N%bs != 0 {
		return torch.Tensor{}, NewFP8GEMMInvalidNError(N, bs)
	}

	numKBlocks := K / bs // matches `k = tl.cdiv(K, BLOCK_SIZE_K)` in Triton

	aData := tensorToFloat32Slice(a, int(M*K))
	bData := tensorToFloat32Slice(b, int(N*K))
	cData := make([]float32, int(M*N))

	// Iterate over K-blocks, accumulating the scaled dot products.
	// This matches the Triton kernel's for-loop over k with:
	//   accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
	for kb := int64(0); kb < numKBlocks; kb++ {
		kStart := kb * bs
		for m := int64(0); m < M; m++ {
			aScale := aS[m*numKBlocks+kb]
			for n := int64(0); n < N; n++ {
				// b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k
				bScale := bS[(n/bs)*numKBlocks+kb]
				var dot float32
				for ki := int64(0); ki < bs; ki++ {
					dot += aData[m*K+kStart+ki] * bData[n*K+kStart+ki]
				}
				cData[m*N+n] += dot * aScale * bScale
			}
		}
	}

	return float32SliceToTensor(cData), nil
}

// tensorToFloat32Slice reads n float32 values out of a gotorch Tensor.
//
// gotorch does not expose a raw data pointer in its public Go API, so we
// extract values element-by-element via torch.Select + Tensor.Item().
// For performance-critical paths, patch gotorch to add a Float32Slice()
// accessor and replace this function.
func tensorToFloat32Slice(t torch.Tensor, n int) []float32 {
	out := make([]float32, n)
	flat := torch.Flatten(t, 0, -1)
	for i := 0; i < n; i++ {
		scalar := flat.Index(int64(i))
		val := scalar.Item()
		switch v := val.(type) {
		case float32:
			out[i] = v
		case float64:
			out[i] = float32(v)
		case int:
			out[i] = float32(v)
		case int32:
			out[i] = float32(v)
		case int64:
			out[i] = float32(v)
		default:
			panic(fmt.Sprintf("unsupported tensor dtype: %T", val))
		}
	}
	return out
}

// float32SliceToTensor wraps a []float32 into a 1-D gotorch Tensor.
func float32SliceToTensor(data []float32) torch.Tensor {
	return torch.NewTensor(data)
}

// MustActQuant is like ActQuant but panics on error.
func MustActQuant(x torch.Tensor, blockSize int, scaleFmt ScaleFmt) ActQuantResult {
	r, err := ActQuant(x, blockSize, scaleFmt)
	if err != nil {
		panic(err)
	}
	return r
}

// MustWeightDequant is like WeightDequant but panics on error.
func MustWeightDequant(x torch.Tensor, s []float32, M, N int64, blockSize int) torch.Tensor {
	r, err := WeightDequant(x, s, M, N, blockSize)
	if err != nil {
		panic(err)
	}
	return r
}

// MustFP8GEMM is like FP8GEMM but panics on error.
func MustFP8GEMM(
	a torch.Tensor, aS []float32,
	b torch.Tensor, bS []float32,
	cfg FP8GEMMConfig,
) torch.Tensor {
	r, err := FP8GEMM(a, aS, b, bS, cfg)
	if err != nil {
		panic(err)
	}
	return r
}
