// Package main of GoSeek-V3 — A Go runtime for DeepSeek models.
//
// This file is a Go rewrite of the original Python implementation
// designed to run DeepSeek models.
// Original by DeepSeek: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
//
// Modified by Lennard <leycm@proton.me>, 2026.
//
// This project is not affiliated with or endorsed by DeepSeek.
package main

import (
	"math"
	"math/rand"

	"goseek/kernel"

	torch "github.com/wangkuiyi/gotorch"
)

var (
	worldSize = 1
	rank      = 0
	blockSize = 128
	gemmImpl  = "bf16"   // "bf16" | "fp8"
	attnImpl  = "absorb" // "naive" | "absorb"
)

// ModelArgs holds all hyperparameters for the Transformer.
type ModelArgs struct {
	MaxBatchSize      int
	MaxSeqLen         int
	DType             string // "bf16" | "fp8"
	ScaleFmt          string // "" | "ue8m0"
	VocabSize         int
	Dim               int
	InterDim          int
	MoEInterDim       int
	NLayers           int
	NDenseLayers      int
	NHeads            int
	NRoutedExperts    int
	NSharedExperts    int
	NActivatedExperts int
	NExpertGroups     int
	NLimitedGroups    int
	ScoreFunc         string // "softmax" | "sigmoid"
	RouteScale        float32
	QLorARank         int
	KVLoRArank        int
	QKNopeHeadDim     int
	QKRopeHeadDim     int
	VHeadDim          int
	OriginalSeqLen    int
	RopeTheta         float32
	RopeFactor        float32
	BetaFast          int
	BetaSlow          int
	MScale            float32
}

// DefaultModelArgs returns the default small-model configuration
// matching the Python ModelArgs defaults.
func DefaultModelArgs() ModelArgs {
	return ModelArgs{
		MaxBatchSize:      8,
		MaxSeqLen:         4096 * 4,
		DType:             "bf16",
		VocabSize:         102400,
		Dim:               2048,
		InterDim:          10944,
		MoEInterDim:       1408,
		NLayers:           27,
		NDenseLayers:      1,
		NHeads:            16,
		NRoutedExperts:    64,
		NSharedExperts:    2,
		NActivatedExperts: 6,
		NExpertGroups:     1,
		NLimitedGroups:    1,
		ScoreFunc:         "softmax",
		RouteScale:        1.0,
		QLorARank:         0,
		KVLoRArank:        512,
		QKNopeHeadDim:     128,
		QKRopeHeadDim:     64,
		VHeadDim:          128,
		OriginalSeqLen:    4096,
		RopeTheta:         10000.0,
		RopeFactor:        40,
		BetaFast:          32,
		BetaSlow:          1,
		MScale:            1.0,
	}
}

// Tensor is a dense, row-major float32 array with an explicit shape.
type Tensor struct {
	Data  []float32
	Shape []int
}

// newTensor allocates a zero-filled Tensor with the given shape.
func newTensor(shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return Tensor{Data: make([]float32, n), Shape: shape}
}

// numel returns the total number of elements.
func (t Tensor) numel() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// clone returns a deep copy.
func (t Tensor) clone() Tensor {
	c := newTensor(t.Shape...)
	copy(c.Data, t.Data)
	return c
}

// toGotorch wraps Data into a 1-D gotorch Tensor for kernel calls.
func (t Tensor) toGotorch() torch.Tensor {
	return torch.NewTensor(t.Data)
}

// fromGotorch reads a flat gotorch Tensor back into a shaped Tensor.
func fromGotorch(gt torch.Tensor, shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	out := Tensor{Data: make([]float32, n), Shape: shape}
	flat := torch.Flatten(gt, 0, -1)
	for i := 0; i < n; i++ {
		switch v := flat.Index(int64(i)).Item().(type) {
		case float32:
			out.Data[i] = v
		case float64:
			out.Data[i] = float32(v)
		}
	}
	return out
}

// randn fills t with small normally-distributed values (useful for tests).
func (t *Tensor) randn() {
	for i := range t.Data {
		t.Data[i] = float32(rand.NormFloat64()) * 0.02
	}
}

// linearFwd computes x @ wT + bias.
// x: (*, inF)  w: (outF, inF)  bias: (outF,) or nil  →  (*, outF)
func linearFwd(x, w Tensor, bias []float32) Tensor {
	outF := w.Shape[0]
	inF := w.Shape[1]
	leading := x.numel() / inF
	out := newTensor(leading, outF)
	for i := 0; i < leading; i++ {
		for j := 0; j < outF; j++ {
			var s float32
			for k := 0; k < inF; k++ {
				s += x.Data[i*inF+k] * w.Data[j*inF+k]
			}
			if bias != nil {
				s += bias[j]
			}
			out.Data[i*outF+j] = s
		}
	}
	outShape := make([]int, len(x.Shape))
	copy(outShape, x.Shape)
	outShape[len(outShape)-1] = outF
	out.Shape = outShape
	return out
}

// linearFwdFP8 uses kernel.FP8GEMM. x and w must already be 2-D.
func linearFwdFP8(x, w Tensor, xScales, wScales []float32, M, N, K int64) Tensor {
	cfg := kernel.FP8GEMMConfig{M: M, N: N, K: K, BlockSize: blockSize}
	gt := kernel.MustFP8GEMM(
		torch.NewTensor(x.Data), xScales,
		torch.NewTensor(w.Data), wScales,
		cfg,
	)
	return fromGotorch(gt, int(M), int(N))
}

// weightDequantTensor wraps kernel.WeightDequant for a Tensor.
func weightDequantTensor(w Tensor, scales []float32, M, N int64) Tensor {
	gt := kernel.MustWeightDequant(torch.NewTensor(w.Data), scales, M, N, blockSize)
	return fromGotorch(gt, int(M), int(N))
}

// rmsNormFwd normalises x along its last dimension using the given weight.
func rmsNormFwd(x Tensor, weight []float32, eps float32) Tensor {
	out := newTensor(x.Shape...)
	dim := x.Shape[len(x.Shape)-1]
	n := x.numel() / dim
	for i := 0; i < n; i++ {
		row := x.Data[i*dim : (i+1)*dim]
		var ss float32
		for _, v := range row {
			ss += v * v
		}
		scale := float32(1.0 / math.Sqrt(float64(ss/float32(dim)+eps)))
		for j, v := range row {
			out.Data[i*dim+j] = v * scale * weight[j]
		}
	}
	return out
}

// softmaxFwd applies softmax along the last dimension (numerically stable).
func softmaxFwd(x Tensor) Tensor {
	out := x.clone()
	dim := x.Shape[len(x.Shape)-1]
	n := x.numel() / dim
	for i := 0; i < n; i++ {
		row := out.Data[i*dim : (i+1)*dim]
		maxV := row[0]
		for _, v := range row {
			if v > maxV {
				maxV = v
			}
		}
		var sum float32
		for j, v := range row {
			e := float32(math.Exp(float64(v - maxV)))
			row[j] = e
			sum += e
		}
		for j := range row {
			row[j] /= sum
		}
	}
	return out
}

// sigmoidFwd applies sigmoid element-wise.
func sigmoidFwd(x Tensor) Tensor {
	out := newTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = float32(1.0 / (1.0 + math.Exp(float64(-v))))
	}
	return out
}

// siluFwd applies SiLU = x * sigmoid(x) element-wise.
func siluFwd(x Tensor) Tensor {
	out := newTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = v * float32(1.0/(1.0+math.Exp(float64(-v))))
	}
	return out
}

// mulElem multiplies two tensors element-wise (same shape required).
func mulElem(a, b Tensor) Tensor {
	out := newTensor(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

// addElem adds two tensors element-wise.
func addElem(a, b Tensor) Tensor {
	out := newTensor(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// mulScalar multiplies every element by s.
func mulScalar(a Tensor, s float32) Tensor {
	out := newTensor(a.Shape...)
	for i, v := range a.Data {
		out.Data[i] = v * s
	}
	return out
}

// reshape returns a view with a new shape (shares the underlying Data slice).
func reshape(t Tensor, shape ...int) Tensor {
	return Tensor{Data: t.Data, Shape: shape}
}

// flatten collapses dims [from, to] (inclusive, negative to = last) into one.
func flattenDims(t Tensor, from, to int) Tensor {
	if to < 0 {
		to = len(t.Shape) - 1
	}
	merged := 1
	for i := from; i <= to; i++ {
		merged *= t.Shape[i]
	}
	ns := make([]int, 0, len(t.Shape)-(to-from))
	ns = append(ns, t.Shape[:from]...)
	ns = append(ns, merged)
	ns = append(ns, t.Shape[to+1:]...)
	return Tensor{Data: t.Data, Shape: ns}
}

// splitLast splits the last dimension of t at position p.
func splitLast(t Tensor, p int) (Tensor, Tensor) {
	n := t.numel() / t.Shape[len(t.Shape)-1]
	d := t.Shape[len(t.Shape)-1]

	shA := make([]int, len(t.Shape))
	copy(shA, t.Shape)
	shA[len(shA)-1] = p

	shB := make([]int, len(t.Shape))
	copy(shB, t.Shape)
	shB[len(shB)-1] = d - p

	a := newTensor(shA...)
	b := newTensor(shB...)
	for i := 0; i < n; i++ {
		copy(a.Data[i*p:], t.Data[i*d:i*d+p])
		copy(b.Data[i*(d-p):], t.Data[i*d+p:(i+1)*d])
	}
	return a, b
}

// concatLast concatenates a and b along their last dimension.
func concatLast(a, b Tensor) Tensor {
	n := a.numel() / a.Shape[len(a.Shape)-1]
	da := a.Shape[len(a.Shape)-1]
	db := b.Shape[len(b.Shape)-1]
	ns := make([]int, len(a.Shape))
	copy(ns, a.Shape)
	ns[len(ns)-1] = da + db
	out := newTensor(ns...)
	for i := 0; i < n; i++ {
		copy(out.Data[i*(da+db):], a.Data[i*da:(i+1)*da])
		copy(out.Data[i*(da+db)+da:], b.Data[i*db:(i+1)*db])
	}
	return out
}

// FreqsCIS holds precomputed cos/sin tables for RoPE.
type FreqsCIS struct {
	Cos []float32 // (maxSeqLen * halfDim)
	Sin []float32
	Seq int
	Dim int // halfDim = QKRopeHeadDim / 2
}

// precomputeFreqsCIS mirrors precompute_freqs_cis from model.py.
func precomputeFreqsCIS(args ModelArgs) FreqsCIS {
	dim := args.QKRopeHeadDim
	seqLen := args.MaxSeqLen
	base := float64(args.RopeTheta)
	factor := float64(args.RopeFactor)
	half := dim / 2

	findCorrDim := func(numRot float64) float64 {
		return float64(dim) * math.Log(float64(args.OriginalSeqLen)/(numRot*2*math.Pi)) /
			(2 * math.Log(base))
	}

	freqs := make([]float32, half)
	for i := 0; i < half; i++ {
		freqs[i] = float32(1.0 / math.Pow(base, float64(2*i)/float64(dim)))
	}

	if seqLen > args.OriginalSeqLen {
		low := int(math.Max(math.Floor(findCorrDim(float64(args.BetaFast))), 0))
		high := int(math.Min(math.Ceil(findCorrDim(float64(args.BetaSlow))), float64(dim-1)))
		for i := 0; i < half; i++ {
			var smooth float32
			switch {
			case i <= low:
				smooth = 1
			case i >= high:
				smooth = 0
			default:
				if low == high {
					smooth = 1
				} else {
					smooth = 1 - float32(i-low)/float32(high-low)
				}
			}
			scaled := float32(float64(freqs[i]) / factor)
			freqs[i] = scaled*(1-smooth) + freqs[i]*smooth
		}
	}

	cosT := make([]float32, seqLen*half)
	sinT := make([]float32, seqLen*half)
	for t := 0; t < seqLen; t++ {
		for j := 0; j < half; j++ {
			angle := float64(t) * float64(freqs[j])
			cosT[t*half+j] = float32(math.Cos(angle))
			sinT[t*half+j] = float32(math.Sin(angle))
		}
	}
	return FreqsCIS{Cos: cosT, Sin: sinT, Seq: seqLen, Dim: half}
}

// applyRotaryEmb applies RoPE to x shaped (bsz, seqLen, nHeads, headDim).
func applyRotaryEmb(x Tensor, fc FreqsCIS, startPos int) Tensor {
	bsz, seqLen, nHeads, headDim := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	half := headDim / 2
	out := newTensor(bsz, seqLen, nHeads, headDim)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			t := startPos + s
			for h := 0; h < nHeads; h++ {
				base := ((b*seqLen+s)*nHeads + h) * headDim
				for j := 0; j < half; j++ {
					cos := fc.Cos[t*fc.Dim+j]
					sin := fc.Sin[t*fc.Dim+j]
					xr := x.Data[base+j]
					xi := x.Data[base+half+j]
					out.Data[base+j] = xr*cos - xi*sin
					out.Data[base+half+j] = xr*sin + xi*cos
				}
			}
		}
	}
	return out
}

// attnScoresNaive: scores[b,s,h,t] = M_d q[b,s,h,d] * k[b,t,h,d]
func attnScoresNaive(q, k Tensor) Tensor {
	bsz, seqLen, nHeads, qkDim := q.Shape[0], q.Shape[1], q.Shape[2], q.Shape[3]
	endPos := k.Shape[1]
	out := newTensor(bsz, seqLen, nHeads, endPos)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				qBase := ((b*seqLen+s)*nHeads + h) * qkDim
				for t := 0; t < endPos; t++ {
					kBase := ((b*endPos+t)*nHeads + h) * qkDim
					var dot float32
					for d := 0; d < qkDim; d++ {
						dot += q.Data[qBase+d] * k.Data[kBase+d]
					}
					out.Data[((b*seqLen+s)*nHeads+h)*endPos+t] = dot
				}
			}
		}
	}
	return out
}

// attnScoresAbsorb computes the absorbed MLA scores in one pass.
// scores[b,s,h,t] = M_c qNope[b,s,h,c]*kv[b,t,c] + M_r qPe[b,s,h,r]*pe[b,t,r]
func attnScoresAbsorb(qNope, qPe, kv, pe Tensor) Tensor {
	bsz, seqLen, nHeads := qNope.Shape[0], qNope.Shape[1], qNope.Shape[2]
	kvRank := kv.Shape[2]
	ropeHd := pe.Shape[2]
	endPos := kv.Shape[1]
	out := newTensor(bsz, seqLen, nHeads, endPos)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				for t := 0; t < endPos; t++ {
					var dot float32
					qnBase := ((b*seqLen+s)*nHeads + h) * kvRank
					kvBase := (b*endPos + t) * kvRank
					for c := 0; c < kvRank; c++ {
						dot += qNope.Data[qnBase+c] * kv.Data[kvBase+c]
					}
					qpBase := ((b*seqLen+s)*nHeads + h) * ropeHd
					peBase := (b*endPos + t) * ropeHd
					for r := 0; r < ropeHd; r++ {
						dot += qPe.Data[qpBase+r] * pe.Data[peBase+r]
					}
					out.Data[((b*seqLen+s)*nHeads+h)*endPos+t] = dot
				}
			}
		}
	}
	return out
}

// applyAttnMask adds -inf above the diagonal (causal mask, prefill only).
func applyAttnMask(scores Tensor, seqLen, endPos int) Tensor {
	out := scores.clone()
	bsz := scores.Shape[0]
	nHeads := scores.Shape[2]
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				for t := s + 1; t < endPos; t++ {
					out.Data[((b*seqLen+s)*nHeads+h)*endPos+t] = float32(math.Inf(-1))
				}
			}
		}
	}
	return out
}

// weightedSumNaive: out[b,s,h,d] = M_t scores[b,s,h,t] * v[b,t,h,d]
func weightedSumNaive(scores, v Tensor) Tensor {
	bsz, seqLen, nHeads, endPos := scores.Shape[0], scores.Shape[1], scores.Shape[2], scores.Shape[3]
	vDim := v.Shape[3]
	out := newTensor(bsz, seqLen, nHeads, vDim)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				for t := 0; t < endPos; t++ {
					w := scores.Data[((b*seqLen+s)*nHeads+h)*endPos+t]
					for d := 0; d < vDim; d++ {
						out.Data[((b*seqLen+s)*nHeads+h)*vDim+d] +=
							w * v.Data[((b*endPos+t)*nHeads+h)*vDim+d]
					}
				}
			}
		}
	}
	return out
}

// absorbQNope: qAbsorbed[b,s,h,c] = M_d qNope[b,s,h,d] * wkvBNope[h,d,c]
func absorbQNope(qNope, wkvBNope Tensor) Tensor {
	bsz, seqLen, nHeads, nopeDim := qNope.Shape[0], qNope.Shape[1], qNope.Shape[2], qNope.Shape[3]
	kvRank := wkvBNope.Shape[2]
	out := newTensor(bsz, seqLen, nHeads, kvRank)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				for c := 0; c < kvRank; c++ {
					var sum float32
					qBase := ((b*seqLen+s)*nHeads + h) * nopeDim
					for d := 0; d < nopeDim; d++ {
						sum += qNope.Data[qBase+d] * wkvBNope.Data[(h*nopeDim+d)*kvRank+c]
					}
					out.Data[((b*seqLen+s)*nHeads+h)*kvRank+c] = sum
				}
			}
		}
	}
	return out
}

// weightedSumAbsorb fuses context accumulation and value projection.
// out[b,s,h,d] = M_c (M_t scores[b,s,h,t]*kv[b,t,c]) * wkvBV[h,d,c]
func weightedSumAbsorb(scores, kv, wkvBV Tensor) Tensor {
	bsz, seqLen, nHeads, endPos := scores.Shape[0], scores.Shape[1], scores.Shape[2], scores.Shape[3]
	kvRank := kv.Shape[2]
	vDim := wkvBV.Shape[1]
	out := newTensor(bsz, seqLen, nHeads, vDim)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < nHeads; h++ {
				ctx := make([]float32, kvRank)
				for t := 0; t < endPos; t++ {
					w := scores.Data[((b*seqLen+s)*nHeads+h)*endPos+t]
					for c := 0; c < kvRank; c++ {
						ctx[c] += w * kv.Data[(b*endPos+t)*kvRank+c]
					}
				}
				for d := 0; d < vDim; d++ {
					var sum float32
					for c := 0; c < kvRank; c++ {
						sum += ctx[c] * wkvBV.Data[(h*vDim+d)*kvRank+c]
					}
					out.Data[((b*seqLen+s)*nHeads+h)*vDim+d] = sum
				}
			}
		}
	}
	return out
}

// RMSNorm layer.
type RMSNorm struct {
	Weight []float32
	Eps    float32
}

func newRMSNorm(dim int) *RMSNorm {
	w := make([]float32, dim)
	for i := range w {
		w[i] = 1.0
	}
	return &RMSNorm{Weight: w, Eps: 1e-6}
}

func (r *RMSNorm) forward(x Tensor) Tensor { return rmsNormFwd(x, r.Weight, r.Eps) }

// LinearLayer holds a weight matrix (outF × inF) and an optional bias.
type LinearLayer struct {
	Weight   Tensor
	Bias     []float32 // nil → no bias
	Scales   []float32 // non-nil → FP8-quantised weight
	ScaleFmt string
}

func newLinearLayer(inF, outF int, bias bool, scaleFmt string) *LinearLayer {
	l := &LinearLayer{Weight: newTensor(outF, inF), ScaleFmt: scaleFmt}
	l.Weight.randn()
	if bias {
		l.Bias = make([]float32, outF)
	}
	return l
}

func (l *LinearLayer) forward(x Tensor) Tensor {
	if l.Scales != nil {
		M := int64(l.Weight.Shape[0])
		N := int64(l.Weight.Shape[1])
		//goland:noinspection GoBoolExpressions
		if gemmImpl == "bf16" {
			dq := weightDequantTensor(l.Weight, l.Scales, M, N)
			return linearFwd(x, dq, l.Bias)
		}

		sfmt := kernel.ScaleFmtFloat32
		if l.ScaleFmt == "ue8m0" {
			sfmt = kernel.ScaleFmtUE8M0
		}

		flat := flattenDims(x, 0, len(x.Shape)-2)
		aq := kernel.MustActQuant(torch.NewTensor(flat.Data), blockSize, sfmt)
		return linearFwdFP8(flat, l.Weight, aq.Scales, l.Scales, int64(flat.Shape[0]), M, N)
	}
	return linearFwd(x, l.Weight, l.Bias)
}

// colParallelLinear partitions outF across worldSize ranks.
type colParallelLinear struct{ *LinearLayer }

func newColParallel(inF, outF int, bias bool, scaleFmt string) *colParallelLinear {
	return &colParallelLinear{newLinearLayer(inF, outF/worldSize, bias, scaleFmt)}
}

// rowParallelLinear partitions inF across worldSize ranks.
type rowParallelLinear struct{ *LinearLayer }

func newRowParallel(inF, outF int, bias bool, scaleFmt string) *rowParallelLinear {
	return &rowParallelLinear{newLinearLayer(inF/worldSize, outF, bias, scaleFmt)}
}

// kvCache holds the cached compressed key/value state.
type kvCache struct {
	// naive impl
	K Tensor // (maxBsz, maxSeq, nLocalHeads, qkHeadDim)
	V Tensor // (maxBsz, maxSeq, nLocalHeads, vHeadDim)
	// absorb impl
	KV Tensor // (maxBsz, maxSeq, kvLoRArank)
	PE Tensor // (maxBsz, maxSeq, qkRopeHeadDim)
}

// MLA is the Multi-Head Latent Attention layer.
type MLA struct {
	args         ModelArgs
	nLocalHeads  int
	qkHeadDim    int
	softmaxScale float32

	wq    *colParallelLinear // used when QLorARank == 0
	wqA   *LinearLayer
	qNorm *RMSNorm
	wqB   *colParallelLinear

	wkvA   *LinearLayer
	kvNorm *RMSNorm
	wkvB   *colParallelLinear

	wo *rowParallelLinear

	cache kvCache
}

func newMLA(args ModelArgs) *MLA {
	m := &MLA{args: args}
	m.nLocalHeads = args.NHeads / worldSize
	m.qkHeadDim = args.QKNopeHeadDim + args.QKRopeHeadDim
	sf := args.ScaleFmt

	if args.QLorARank == 0 {
		m.wq = newColParallel(args.Dim, args.NHeads*m.qkHeadDim, false, sf)
	} else {
		m.wqA = newLinearLayer(args.Dim, args.QLorARank, false, sf)
		m.qNorm = newRMSNorm(args.QLorARank)
		m.wqB = newColParallel(args.QLorARank, args.NHeads*m.qkHeadDim, false, sf)
	}
	m.wkvA = newLinearLayer(args.Dim, args.KVLoRArank+args.QKRopeHeadDim, false, sf)
	m.kvNorm = newRMSNorm(args.KVLoRArank)
	m.wkvB = newColParallel(args.KVLoRArank, args.NHeads*(args.QKNopeHeadDim+args.VHeadDim), false, sf)
	m.wo = newRowParallel(args.NHeads*args.VHeadDim, args.Dim, false, sf)

	m.softmaxScale = float32(math.Pow(float64(m.qkHeadDim), -0.5))
	if args.MaxSeqLen > args.OriginalSeqLen {
		ms := 0.1*float64(args.MScale)*math.Log(float64(args.RopeFactor)) + 1.0
		m.softmaxScale = float32(float64(m.softmaxScale) * ms * ms)
	}

	bsz, seq, nh := args.MaxBatchSize, args.MaxSeqLen, m.nLocalHeads
	//goland:noinspection GoBoolExpressions
	if attnImpl == "naive" {
		m.cache.K = newTensor(bsz, seq, nh, m.qkHeadDim)
		m.cache.V = newTensor(bsz, seq, nh, args.VHeadDim)
	} else {
		m.cache.KV = newTensor(bsz, seq, args.KVLoRArank)
		m.cache.PE = newTensor(bsz, seq, args.QKRopeHeadDim)
	}
	return m
}

func (m *MLA) forward(x Tensor, startPos int, fc FreqsCIS, mask bool) Tensor {
	bsz, seqLen := x.Shape[0], x.Shape[1]
	endPos := startPos + seqLen

	// query
	var qRaw Tensor
	if m.args.QLorARank == 0 {
		qRaw = m.wq.forward(x)
	} else {
		qRaw = m.wqB.forward(m.qNorm.forward(m.wqA.forward(x)))
	}
	q := reshape(qRaw, bsz, seqLen, m.nLocalHeads, m.qkHeadDim)
	qNope, qPe := splitLast(q, m.args.QKNopeHeadDim)
	qPe = applyRotaryEmb(qPe, fc, startPos)

	// compressed kv
	kvRaw := m.wkvA.forward(x)
	kvPart, kPeRaw := splitLast(kvRaw, m.args.KVLoRArank)
	kPe := applyRotaryEmb(reshape(kPeRaw, bsz, seqLen, 1, m.args.QKRopeHeadDim), fc, startPos)
	kPe = reshape(kPe, bsz, seqLen, m.args.QKRopeHeadDim)

	var scores Tensor

	//goland:noinspection GoBoolExpressions
	if attnImpl == "naive" {
		q = concatLast(qNope, qPe)

		kvFull := reshape(
			m.wkvB.forward(m.kvNorm.forward(kvPart)),
			bsz, seqLen, m.nLocalHeads, m.args.QKNopeHeadDim+m.args.VHeadDim,
		)
		kNope, v := splitLast(kvFull, m.args.QKNopeHeadDim)

		// broadcast kPe over heads
		kPeExp := newTensor(bsz, seqLen, m.nLocalHeads, m.args.QKRopeHeadDim)
		for b := 0; b < bsz; b++ {
			for s := 0; s < seqLen; s++ {
				for h := 0; h < m.nLocalHeads; h++ {
					src := kPe.Data[(b*seqLen+s)*m.args.QKRopeHeadDim:]
					dst := kPeExp.Data[((b*seqLen+s)*m.nLocalHeads+h)*m.args.QKRopeHeadDim:]
					copy(dst[:m.args.QKRopeHeadDim], src[:m.args.QKRopeHeadDim])
				}
			}
		}
		k := concatLast(kNope, kPeExp)

		for b := 0; b < bsz; b++ {
			for s := 0; s < seqLen; s++ {
				pos := startPos + s
				for h := 0; h < m.nLocalHeads; h++ {
					copy(
						m.cache.K.Data[((b*m.args.MaxSeqLen+pos)*m.nLocalHeads+h)*m.qkHeadDim:],
						k.Data[((b*seqLen+s)*m.nLocalHeads+h)*m.qkHeadDim:(((b*seqLen+s)*m.nLocalHeads+h)+1)*m.qkHeadDim],
					)
					copy(
						m.cache.V.Data[((b*m.args.MaxSeqLen+pos)*m.nLocalHeads+h)*m.args.VHeadDim:],
						v.Data[((b*seqLen+s)*m.nLocalHeads+h)*m.args.VHeadDim:(((b*seqLen+s)*m.nLocalHeads+h)+1)*m.args.VHeadDim],
					)
				}
			}
		}

		kCached := newTensor(bsz, endPos, m.nLocalHeads, m.qkHeadDim)
		vCached := newTensor(bsz, endPos, m.nLocalHeads, m.args.VHeadDim)
		for b := 0; b < bsz; b++ {
			copy(kCached.Data[b*endPos*m.nLocalHeads*m.qkHeadDim:],
				m.cache.K.Data[b*m.args.MaxSeqLen*m.nLocalHeads*m.qkHeadDim:b*m.args.MaxSeqLen*m.nLocalHeads*m.qkHeadDim+endPos*m.nLocalHeads*m.qkHeadDim])
			copy(vCached.Data[b*endPos*m.nLocalHeads*m.args.VHeadDim:],
				m.cache.V.Data[b*m.args.MaxSeqLen*m.nLocalHeads*m.args.VHeadDim:b*m.args.MaxSeqLen*m.nLocalHeads*m.args.VHeadDim+endPos*m.nLocalHeads*m.args.VHeadDim])
		}

		scores = mulScalar(attnScoresNaive(q, kCached), m.softmaxScale)
		if mask {
			scores = applyAttnMask(scores, seqLen, endPos)
		}
		scores = softmaxFwd(scores)
		x = weightedSumNaive(scores, vCached)

	} else {
		// absorb path: split wkvB weight into nope and value sub-matrices
		nopeDim := m.args.QKNopeHeadDim
		vDim := m.args.VHeadDim
		kvRank := m.args.KVLoRArank
		nh := m.nLocalHeads

		// wkvB.Weight shape: (nh*(nopeDim+vDim), kvRank)
		// wkvBNope[h, d, c] and wkvBV[h, d, c]
		wkvBNope := newTensor(nh, nopeDim, kvRank)
		wkvBV := newTensor(nh, vDim, kvRank)
		for h := 0; h < nh; h++ {
			for d := 0; d < nopeDim; d++ {
				copy(
					wkvBNope.Data[(h*nopeDim+d)*kvRank:],
					m.wkvB.Weight.Data[(h*(nopeDim+vDim)+d)*kvRank:(h*(nopeDim+vDim)+d+1)*kvRank],
				)
			}
			for d := 0; d < vDim; d++ {
				copy(
					wkvBV.Data[(h*vDim+d)*kvRank:],
					m.wkvB.Weight.Data[(h*(nopeDim+vDim)+nopeDim+d)*kvRank:(h*(nopeDim+vDim)+nopeDim+d+1)*kvRank],
				)
			}
		}

		qAbsorbed := absorbQNope(qNope, wkvBNope)
		kvNormed := m.kvNorm.forward(kvPart)

		for b := 0; b < bsz; b++ {
			for s := 0; s < seqLen; s++ {
				pos := startPos + s
				copy(
					m.cache.KV.Data[(b*m.args.MaxSeqLen+pos)*kvRank:],
					kvNormed.Data[(b*seqLen+s)*kvRank:(b*seqLen+s+1)*kvRank],
				)
				copy(
					m.cache.PE.Data[(b*m.args.MaxSeqLen+pos)*m.args.QKRopeHeadDim:],
					kPe.Data[(b*seqLen+s)*m.args.QKRopeHeadDim:(b*seqLen+s+1)*m.args.QKRopeHeadDim],
				)
			}
		}

		kvCached := newTensor(bsz, endPos, kvRank)
		peCached := newTensor(bsz, endPos, m.args.QKRopeHeadDim)
		for b := 0; b < bsz; b++ {
			copy(kvCached.Data[b*endPos*kvRank:],
				m.cache.KV.Data[b*m.args.MaxSeqLen*kvRank:b*m.args.MaxSeqLen*kvRank+endPos*kvRank])
			copy(peCached.Data[b*endPos*m.args.QKRopeHeadDim:],
				m.cache.PE.Data[b*m.args.MaxSeqLen*m.args.QKRopeHeadDim:b*m.args.MaxSeqLen*m.args.QKRopeHeadDim+endPos*m.args.QKRopeHeadDim])
		}

		scores = mulScalar(attnScoresAbsorb(qAbsorbed, qPe, kvCached, peCached), m.softmaxScale)
		if mask {
			scores = applyAttnMask(scores, seqLen, endPos)
		}
		scores = softmaxFwd(scores)
		x = weightedSumAbsorb(scores, kvCached, wkvBV)
	}

	x = reshape(x, bsz*seqLen, m.nLocalHeads*m.args.VHeadDim)
	x = m.wo.forward(x)
	return reshape(x, bsz, seqLen, m.args.Dim)
}

// MLP is the SwiGLU feed-forward sublayer.
type MLP struct {
	W1 *colParallelLinear
	W2 *rowParallelLinear
	W3 *colParallelLinear
}

func newMLP(dim, interDim int, scaleFmt string) *MLP {
	return &MLP{
		W1: newColParallel(dim, interDim, false, scaleFmt),
		W2: newRowParallel(interDim, dim, false, scaleFmt),
		W3: newColParallel(dim, interDim, false, scaleFmt),
	}
}

func (m *MLP) forward(x Tensor) Tensor {
	return m.W2.forward(mulElem(siluFwd(m.W1.forward(x)), m.W3.forward(x)))
}

// Expert is a single MoE expert with the same SwiGLU topology as MLP,
// but uses plain (non-parallel) LinearLayers.
type Expert struct {
	W1, W2, W3 *LinearLayer
}

func newExpert(dim, interDim int, scaleFmt string) *Expert {
	return &Expert{
		W1: newLinearLayer(dim, interDim, false, scaleFmt),
		W2: newLinearLayer(interDim, dim, false, scaleFmt),
		W3: newLinearLayer(dim, interDim, false, scaleFmt),
	}
}

func (e *Expert) forward(x Tensor) Tensor {
	return e.W2.forward(mulElem(siluFwd(e.W1.forward(x)), e.W3.forward(x)))
}

// Gate routes tokens to the top-k experts.
type Gate struct {
	args   ModelArgs
	Weight LinearLayer
	Bias   []float32 // only allocated when Dim == 7168
}

func newGate(args ModelArgs) *Gate {
	g := &Gate{args: args, Weight: *newLinearLayer(args.Dim, args.NRoutedExperts, false, "")}
	if args.Dim == 7168 {
		g.Bias = make([]float32, args.NRoutedExperts)
	}
	return g
}

// topKIdx returns the indices of the k largest values in row.
func topKIdx(row []float32, k int) []int {
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if row[idx[j]] > row[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	return idx[:k]
}

// forward returns (weights Tensor (nTokens, k), indices [][]int (nTokens, k)).
func (g *Gate) forward(x Tensor) (Tensor, [][]int) {
	nTokens := x.Shape[0]
	nExperts := g.args.NRoutedExperts
	k := g.args.NActivatedExperts

	scores := g.Weight.forward(x) // (nTokens, nExperts)
	if g.args.ScoreFunc == "softmax" {
		scores = softmaxFwd(scores)
	} else {
		scores = sigmoidFwd(scores)
	}
	origScores := scores.clone()

	if g.Bias != nil {
		for i := 0; i < nTokens; i++ {
			for j := 0; j < nExperts; j++ {
				scores.Data[i*nExperts+j] += g.Bias[j]
			}
		}
	}

	if g.args.NExpertGroups > 1 {
		ng := g.args.NExpertGroups
		perGroup := nExperts / ng
		for i := 0; i < nTokens; i++ {
			groupScores := make([]float32, ng)
			for gr := 0; gr < ng; gr++ {
				seg := scores.Data[i*nExperts+gr*perGroup : i*nExperts+(gr+1)*perGroup]
				if g.Bias == nil {
					for _, v := range seg {
						if v > groupScores[gr] {
							groupScores[gr] = v
						}
					}
				} else {
					top2 := topKIdx(seg, 2)
					groupScores[gr] = seg[top2[0]] + seg[top2[1]]
				}
			}
			allowed := topKIdx(groupScores, g.args.NLimitedGroups)
			allowedSet := make(map[int]bool, len(allowed))
			for _, gr := range allowed {
				allowedSet[gr] = true
			}
			for gr := 0; gr < ng; gr++ {
				if !allowedSet[gr] {
					for e := 0; e < perGroup; e++ {
						scores.Data[i*nExperts+gr*perGroup+e] = float32(math.Inf(-1))
					}
				}
			}
		}
	}

	indices := make([][]int, nTokens)
	weights := newTensor(nTokens, k)
	for i := 0; i < nTokens; i++ {
		row := make([]float32, nExperts)
		copy(row, scores.Data[i*nExperts:(i+1)*nExperts])
		topIdx := topKIdx(row, k)
		indices[i] = topIdx
		for j, ei := range topIdx {
			weights.Data[i*k+j] = origScores.Data[i*nExperts+ei]
		}
	}

	if g.args.ScoreFunc == "sigmoid" {
		for i := 0; i < nTokens; i++ {
			var sum float32
			for j := 0; j < k; j++ {
				sum += weights.Data[i*k+j]
			}
			for j := 0; j < k; j++ {
				weights.Data[i*k+j] /= sum
			}
		}
	}

	rs := float32(g.args.RouteScale)
	for i := range weights.Data {
		weights.Data[i] *= rs
	}
	return weights, indices
}

// MoE is the Mixture-of-Experts feed-forward sublayer.
type MoE struct {
	args           ModelArgs
	expertStartIdx int
	expertEndIdx   int
	Gate           *Gate
	Experts        []*Expert // nil slots for non-local experts
	SharedExperts  *MLP
}

func newMoE(args ModelArgs) *MoE {
	nLocal := args.NRoutedExperts / worldSize
	start := rank * nLocal
	end := start + nLocal
	sf := args.ScaleFmt

	experts := make([]*Expert, args.NRoutedExperts)
	for i := start; i < end; i++ {
		experts[i] = newExpert(args.Dim, args.MoEInterDim, sf)
	}
	return &MoE{
		args:           args,
		expertStartIdx: start,
		expertEndIdx:   end,
		Gate:           newGate(args),
		Experts:        experts,
		SharedExperts:  newMLP(args.Dim, args.NSharedExperts*args.MoEInterDim, sf),
	}
}

func (m *MoE) forward(x Tensor) Tensor {
	origShape := x.Shape
	dim := origShape[len(origShape)-1]
	nTokens := x.numel() / dim
	flat := reshape(x, nTokens, dim)

	weights, indices := m.Gate.forward(flat)
	y := newTensor(nTokens, dim)

	counts := make([]int, m.args.NRoutedExperts)
	for _, row := range indices {
		for _, ei := range row {
			counts[ei]++
		}
	}

	for ei := m.expertStartIdx; ei < m.expertEndIdx; ei++ {
		if counts[ei] == 0 {
			continue
		}
		var tokenRows, topPositions []int
		for i, row := range indices {
			for j, e := range row {
				if e == ei {
					tokenRows = append(tokenRows, i)
					topPositions = append(topPositions, j)
				}
			}
		}

		batch := newTensor(len(tokenRows), dim)
		for bi, ti := range tokenRows {
			copy(batch.Data[bi*dim:], flat.Data[ti*dim:(ti+1)*dim])
		}

		out := m.Experts[ei].forward(batch)
		for bi, ti := range tokenRows {
			w := weights.Data[ti*m.args.NActivatedExperts+topPositions[bi]]
			for d := 0; d < dim; d++ {
				y.Data[ti*dim+d] += out.Data[bi*dim+d] * w
			}
		}
	}

	z := m.SharedExperts.forward(flat)
	return reshape(addElem(y, z), origShape...)
}

// ffnLayer is the common gointerface for MLP and MoE.
type ffnLayer interface{ forward(Tensor) Tensor }

// Block is a single Transformer layer (pre-norm, residual).
type Block struct {
	Attn     *MLA
	FFN      ffnLayer
	AttnNorm *RMSNorm
	FFNNorm  *RMSNorm
}

func newBlock(layerID int, args ModelArgs) *Block {
	b := &Block{
		Attn:     newMLA(args),
		AttnNorm: newRMSNorm(args.Dim),
		FFNNorm:  newRMSNorm(args.Dim),
	}
	if layerID < args.NDenseLayers {
		b.FFN = newMLP(args.Dim, args.InterDim, args.ScaleFmt)
	} else {
		b.FFN = newMoE(args)
	}
	return b
}

func (b *Block) forward(x Tensor, startPos int, fc FreqsCIS, mask bool) Tensor {
	x = addElem(x, b.Attn.forward(b.AttnNorm.forward(x), startPos, fc, mask))
	x = addElem(x, b.FFN.forward(b.FFNNorm.forward(x)))
	return x
}

// ParallelEmbedding shards the vocabulary across worldSize ranks.
type ParallelEmbedding struct {
	Weight        Tensor
	VocabStartIdx int
	VocabEndIdx   int
	Dim           int
}

func newParallelEmbedding(vocabSize, dim int) *ParallelEmbedding {
	part := vocabSize / worldSize
	e := &ParallelEmbedding{
		Weight:        newTensor(part, dim),
		VocabStartIdx: rank * part,
		VocabEndIdx:   rank*part + part,
		Dim:           dim,
	}
	e.Weight.randn()
	return e
}

func (e *ParallelEmbedding) forward(tokens [][]int) Tensor {
	bsz := len(tokens)
	seqLen := len(tokens[0])
	out := newTensor(bsz, seqLen, e.Dim)
	for b := 0; b < bsz; b++ {
		for s := 0; s < seqLen; s++ {
			id := tokens[b][s]
			//goland:noinspection GoBoolExpressions
			if worldSize > 1 && (id < e.VocabStartIdx || id >= e.VocabEndIdx) {
				continue // zero contribution; all_reduce fills it in
			}
			local := id - e.VocabStartIdx
			copy(
				out.Data[(b*seqLen+s)*e.Dim:],
				e.Weight.Data[local*e.Dim:(local+1)*e.Dim],
			)
		}
	}
	return out
}

// Transformer is the full autoregressive language model.
type Transformer struct {
	args     ModelArgs
	Embed    *ParallelEmbedding
	Layers   []*Block
	Norm     *RMSNorm
	Head     *colParallelLinear
	FreqsCIS FreqsCIS
}

// NewTransformer constructs and randomly initializes a Transformer.
func NewTransformer(args ModelArgs) *Transformer {
	t := &Transformer{args: args}
	t.Embed = newParallelEmbedding(args.VocabSize, args.Dim)
	t.Layers = make([]*Block, args.NLayers)
	for i := range t.Layers {
		t.Layers[i] = newBlock(i, args)
	}
	t.Norm = newRMSNorm(args.Dim)
	t.Head = newColParallel(args.Dim, args.VocabSize, false, "")
	t.FreqsCIS = precomputeFreqsCIS(args)
	return t
}

// Forward runs the model and returns logits shaped (bsz, vocabSize/worldSize).
// tokens is (bsz, seqLen).
func (t *Transformer) Forward(tokens [][]int, startPos int) Tensor {
	seqLen := len(tokens[0])
	h := t.Embed.forward(tokens)

	mask := seqLen > 1
	for _, layer := range t.Layers {
		h = layer.forward(h, startPos, t.FreqsCIS, mask)
	}

	// take last token position: (bsz, dim)
	bsz := len(tokens)
	last := newTensor(bsz, t.args.Dim)
	for b := 0; b < bsz; b++ {
		copy(last.Data[b*t.args.Dim:],
			h.Data[(b*seqLen+(seqLen-1))*t.args.Dim:(b*seqLen+seqLen)*t.args.Dim])
	}

	last = t.Norm.forward(last)
	logits := t.Head.forward(last) // (bsz, vocabSize/worldSize)

	// distributed: all_gather across ranks + concatenate
	return logits
}
