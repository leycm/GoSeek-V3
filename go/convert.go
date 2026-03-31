// Package main of GoSeek-V3 — A Go runtime for DeepSeek models.
//
// This file is a Go rewrite of the original Python implementation
// designed to run DeepSeek models.
// Original by DeepSeek: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/convert.py
//
// Modified by Lennard <leycm@proton.me>, 2026.
//
// This project is not affiliated with or endorsed by DeepSeek.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// mapping mirrors the Python `mapping` dict:
// "hf_key": ("new_key", split_dim)   split_dim == -1 means no split
var mapping = map[string]struct {
	newKey string
	dim    int // -1 = no split
}{
	"embed_tokens":             {"embed", 0},
	"input_layernorm":          {"attn_norm", -1},
	"post_attention_layernorm": {"ffn_norm", -1},
	"q_proj":                   {"wq", 0},
	"q_a_proj":                 {"wq_a", -1},
	"q_a_layernorm":            {"q_norm", -1},
	"q_b_proj":                 {"wq_b", 0},
	"kv_a_proj_with_mqa":       {"wkv_a", -1},
	"kv_a_layernorm":           {"kv_norm", -1},
	"kv_b_proj":                {"wkv_b", 0},
	"o_proj":                   {"wo", 1},
	"gate":                     {"gate", -1},
	"gate_proj":                {"w1", 0},
	"down_proj":                {"w2", 1},
	"up_proj":                  {"w3", 0},
	"norm":                     {"norm", -1},
	"lm_head":                  {"head", 0},
	"scale":                    {"scale", -1},
}

// dtype constants (subset of safetensors spec used by DeepSeek weights).
type dtype string

const (
	dtypeBF16 dtype = "BF16"
	dtypeF32  dtype = "F32"
	dtypeF8E4 dtype = "F8_E4M3"
	dtypeI32  dtype = "I32"
)

func dtypeBytes(d dtype) int {
	switch d {
	case dtypeF32, dtypeI32:
		return 4
	case dtypeBF16:
		return 2
	case dtypeF8E4:
		return 1
	default:
		return 0
	}
}

// TensorMeta describes one tensor inside a safetensors file.
type TensorMeta struct {
	DType   dtype    `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"` // byte range inside the data section
}

// safetensorsHeader is the JSON structure at the start of a .safetensors file.
type safetensorsHeader map[string]*TensorMeta // name → meta; "__metadata__" key is ignored

// RawTensor holds a tensor's metadata and its raw bytes.
type RawTensor struct {
	Meta TensorMeta
	Data []byte
}

// numel returns the total number of elements.
func (t *RawTensor) numel() int64 {
	n := int64(1)
	for _, d := range t.Meta.Shape {
		n *= d
	}
	return n
}

// clone returns a deep copy.
func (t *RawTensor) clone() *RawTensor {
	c := &RawTensor{Meta: t.Meta}
	c.Meta.Shape = append([]int64(nil), t.Meta.Shape...)
	c.Data = append([]byte(nil), t.Data...)
	return c
}

// narrow returns a view that slices dimension `dim` to [start, start+size).
// Only contiguous slicing along dimension 0 or 1 (for 2-D tensors) is needed.
func (t *RawTensor) narrow(dim int, start, size int64) *RawTensor {
	elemBytes := int64(dtypeBytes(t.Meta.DType))
	shape := t.Meta.Shape

	// Compute stride for `dim` (row-major).
	stride := int64(1)
	for i := dim + 1; i < len(shape); i++ {
		stride *= shape[i]
	}

	// Number of "outer" slices (all dims before `dim`).
	outer := int64(1)
	for i := 0; i < dim; i++ {
		outer *= shape[i]
	}

	srcStride := shape[dim] * stride * elemBytes
	dstStride := size * stride * elemBytes
	out := make([]byte, outer*dstStride)

	for o := int64(0); o < outer; o++ {
		srcOff := o*srcStride + start*stride*elemBytes
		dstOff := o * dstStride
		copy(out[dstOff:dstOff+dstStride], t.Data[srcOff:srcOff+dstStride])
	}

	newShape := append([]int64(nil), shape...)
	newShape[dim] = size
	return &RawTensor{
		Meta: TensorMeta{DType: t.Meta.DType, Shape: newShape},
		Data: out,
	}
}

// readSafetensors reads all tensors from a .safetensors file into memory.
func readSafetensors(path string) (map[string]*RawTensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// First 8 bytes: little-endian uint64 = header length.
	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("reading header length: %w", err)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}

	var header safetensorsHeader
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("parsing header JSON: %w", err)
	}

	// dataStart is the offset of the data section from the beginning of the file.
	dataStart := int64(8 + headerLen)

	result := make(map[string]*RawTensor, len(header))
	for name, meta := range header {
		if name == "__metadata__" || meta == nil {
			continue
		}
		lo := meta.Offsets[0]
		hi := meta.Offsets[1]
		data := make([]byte, hi-lo)
		if _, err := f.ReadAt(data, dataStart+lo); err != nil {
			return nil, fmt.Errorf("reading tensor %q: %w", name, err)
		}
		result[name] = &RawTensor{Meta: *meta, Data: data}
	}
	return result, nil
}

// writeSafetensors serialises tensors to a .safetensors file.
func writeSafetensors(path string, tensors map[string]*RawTensor) error {
	// Sort names for deterministic output.
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)

	// Build header.
	type headerEntry struct {
		DType   dtype    `json:"dtype"`
		Shape   []int64  `json:"shape"`
		Offsets [2]int64 `json:"data_offsets"`
	}
	hdr := make(map[string]headerEntry, len(tensors))
	var offset int64
	for _, name := range names {
		t := tensors[name]
		size := int64(len(t.Data))
		hdr[name] = headerEntry{
			DType:   t.Meta.DType,
			Shape:   t.Meta.Shape,
			Offsets: [2]int64{offset, offset + size},
		}
		offset += size
	}

	headerBytes, err := json.Marshal(hdr)
	if err != nil {
		return err
	}

	// Pad header to 8-byte boundary (spec recommendation).
	for len(headerBytes)%8 != 0 {
		headerBytes = append(headerBytes, ' ')
	}

	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()

	if err := binary.Write(out, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return err
	}
	if _, err := out.Write(headerBytes); err != nil {
		return err
	}
	for _, name := range names {
		if _, err := out.Write(tensors[name].Data); err != nil {
			return err
		}
	}
	return nil
}

func rewriteName(name string) (newName string, mapEntry struct {
	newKey string
	dim    int
}, err error) {
	// Strip "model." prefix.
	name = strings.TrimPrefix(name, "model.")

	name = strings.ReplaceAll(name, "self_attn", "attn")
	name = strings.ReplaceAll(name, "mlp", "ffn")
	name = strings.ReplaceAll(name, "weight_scale_inv", "scale")
	name = strings.ReplaceAll(name, "e_score_correction_bias", "bias")

	parts := strings.Split(name, ".")
	if len(parts) < 2 {
		return "", mapEntry, fmt.Errorf("unexpected name %q: too few parts", name)
	}
	key := parts[len(parts)-2]

	entry, ok := mapping[key]
	if !ok {
		return "", mapEntry, fmt.Errorf("key %q not found in mapping", key)
	}

	parts[len(parts)-2] = entry.newKey
	return strings.Join(parts, "."), entry, nil
}

// expertIndex extracts the expert index from a name like
// "layers.3.ffn.experts.42.w1.weight", returning -1 if not an expert tensor.
func expertIndex(name string) int {
	parts := strings.Split(name, ".")
	for i, p := range parts {
		if p == "experts" && i+1 < len(parts) {
			if idx, err := strconv.Atoi(parts[i+1]); err == nil {
				return idx
			}
		}
	}
	return -1
}

func isSharedExpert(name string) bool {
	return strings.Contains(name, "shared_experts")
}

func progressBar(current, total int, label string) {
	const width = 40
	pct := float64(current) / float64(total)
	filled := int(math.Round(pct * width))
	bar := strings.Repeat("=", filled) + strings.Repeat("-", width-filled)
	fmt.Printf("\r%s [%s] %d/%d", label, bar, current, total)
	if current == total {
		fmt.Println()
	}
}

func convertCheckpoint(hfCkptPath, savePath string, nExperts, mp int) error {
	nLocalExperts := nExperts / mp

	// stateDicts[i] holds tensors destined for rank i.
	stateDicts := make([]map[string]*RawTensor, mp)
	for i := range stateDicts {
		stateDicts[i] = make(map[string]*RawTensor)
	}

	// Collect all .safetensors files.
	pattern := filepath.Join(hfCkptPath, "*.safetensors")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	sort.Strings(files)
	if len(files) == 0 {
		return fmt.Errorf("no .safetensors files found in %s", hfCkptPath)
	}

	fmt.Printf("Converting %d checkpoint shard(s)...\n", len(files))

	for fi, filePath := range files {
		progressBar(fi, len(files), "Loading")

		tensors, err := readSafetensors(filePath)
		if err != nil {
			return fmt.Errorf("reading %s: %w", filePath, err)
		}

		for origName, param := range tensors {
			// Skip MTP module layer 61.
			if strings.Contains(origName, "model.layers.61") {
				continue
			}

			newName, entry, err := rewriteName(origName)
			if err != nil {
				return fmt.Errorf("rewriting name %q: %w", origName, err)
			}

			for i := 0; i < mp; i++ {
				if !isSharedExpert(newName) {
					idx := expertIndex(newName)
					if idx >= 0 {
						// Routed expert: only include in the rank that owns it.
						lo := i * nLocalExperts
						hi := (i + 1) * nLocalExperts
						if idx < lo || idx >= hi {
							continue
						}
						stateDicts[i][newName] = param.clone()
						continue
					}
				}

				// Non-expert (or shared expert): optionally shard along `dim`.
				if entry.dim < 0 {
					// No sharding — all ranks get the same tensor.
					stateDicts[i][newName] = param.clone()
				} else {
					dim := entry.dim
					total := param.Meta.Shape[dim]
					if total%int64(mp) != 0 {
						return fmt.Errorf(
							"dimension %d of %q (size %d) is not divisible by mp=%d",
							dim, origName, total, mp,
						)
					}
					shardSize := total / int64(mp)
					shard := param.narrow(dim, int64(i)*shardSize, shardSize)
					stateDicts[i][newName] = shard
				}
			}
		}
	}
	progressBar(len(files), len(files), "Loading")

	// Write output files.
	if err := os.MkdirAll(savePath, 0o755); err != nil {
		return err
	}

	fmt.Printf("Writing %d rank file(s)...\n", mp)
	for i := 0; i < mp; i++ {
		progressBar(i, mp, "Saving ")
		outFile := filepath.Join(savePath, fmt.Sprintf("model%d-mp%d.safetensors", i, mp))
		if err := writeSafetensors(outFile, stateDicts[i]); err != nil {
			return fmt.Errorf("writing rank %d: %w", i, err)
		}
	}
	progressBar(mp, mp, "Saving ")

	// Copy tokenizer files (anything with "token" in the name).
	entries, err := os.ReadDir(hfCkptPath)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if strings.Contains(e.Name(), "token") {
			src := filepath.Join(hfCkptPath, e.Name())
			dst := filepath.Join(savePath, e.Name())
			if err := copyFile(src, dst); err != nil {
				fmt.Fprintf(os.Stderr, "warning: could not copy %s: %v\n", e.Name(), err)
			}
		}
	}

	fmt.Println("Done.")
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

func main() {
	hfCkptPath := flag.String("hf-ckpt-path", "", "Path to the HuggingFace checkpoint directory (required)")
	savePath := flag.String("save-path", "", "Path to save converted checkpoint files (required)")
	nExperts := flag.Int("n-experts", 0, "Total number of experts in the model (required)")
	mp := flag.Int("model-parallel", 0, "Model parallelism factor (required)")
	flag.Parse()

	if *hfCkptPath == "" || *savePath == "" || *nExperts == 0 || *mp == 0 {
		fmt.Fprintln(os.Stderr, "All flags are required: --hf-ckpt-path, --save-path, --n-experts, --model-parallel")
		flag.Usage()
		os.Exit(1)
	}
	if *nExperts%*mp != 0 {
		fmt.Fprintf(os.Stderr, "n-experts (%d) must be divisible by model-parallel (%d)\n", *nExperts, *mp)
		os.Exit(1)
	}

	if err := convertCheckpoint(*hfCkptPath, *savePath, *nExperts, *mp); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}
