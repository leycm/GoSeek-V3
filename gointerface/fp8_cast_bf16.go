// Package kernel of GoSeek-V3 — A Go runtime for DeepSeek models.
//
// This file is a Go rewrite of the original Python implementation
// designed to run DeepSeek models.
// Original by DeepSeek: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.go
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
	"math"
	"os"
	"path/filepath"
	"sort"

	"goseek/kernel"

	"github.com/nlpodyssey/safetensors"
	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	inputPath := flag.String("input-fp8-hf-path", "", "Path to FP8 weights and model index")
	outputPath := flag.String("output-bf16-hf-path", "", "Path to save converted BF16 weights")
	flag.Parse()

	if *inputPath == "" || *outputPath == "" {
		fmt.Println("Both input and output paths are required")
		return
	}

	if err := ConvertFP8ToBF16(*inputPath, *outputPath); err != nil {
		fmt.Println("Error:", err)
	}
}

func ConvertFP8ToBF16(fp8Path, bf16Path string) error {
	if err := os.MkdirAll(bf16Path, 0755); err != nil {
		return err
	}

	// Load model index
	modelIndexFile := filepath.Join(fp8Path, "model.safetensors.index.json")
	data, err := os.ReadFile(modelIndexFile)
	if err != nil {
		return err
	}

	var modelIndex struct {
		Metadata  map[string]interface{} `json:"metadata"`
		WeightMap map[string]string      `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &modelIndex); err != nil {
		return err
	}
	weightMap := modelIndex.WeightMap
	var fp8WeightNames []string

	safetensorFiles, err := filepath.Glob(filepath.Join(fp8Path, "*.safetensors"))
	if err != nil {
		return err
	}

	sort.Strings(safetensorFiles)

	for _, file := range safetensorFiles {
		bytesData, err := os.ReadFile(file)
		if err != nil {
			return err
		}

		st, err := safetensors.Deserialize(bytesData)
		if err != nil {
			return err
		}

		newTensors := make(map[string]safetensors.TensorView)
		for _, name := range st.Names() {
			tensor, ok := st.Tensor(name)
			if !ok {
				continue
			}
			if len(name) >= 10 && name[len(name)-10:] == "_scale_inv" {
				continue // skip scale_inv
			}

			// FP8 check: 1-byte tensor
			if tensor.DType() == safetensors.U8 {
				scaleName := name + "_scale_inv"
				scaleTensor, ok := st.Tensor(scaleName)
				if !ok {
					fmt.Printf("Warning: missing scale_inv for %s, skipping\n", name)
					newTensors[name] = tensor
					continue
				}
				fp8WeightNames = append(fp8WeightNames, name)

				// Convert data bytes to float32 tensor
				M := int64(tensor.Shape()[0])
				N := int64(tensor.Shape()[1])
				x := bytesToTorchTensor(tensor.Data(), M, N)
				s := bytesToFloat32Slice(scaleTensor.Data())
				dequant := kernel.MustWeightDequant(x, s, M, N, 128)

				// Convert back to bytes
				newTensors[name] = tensorFromTorch(dequant, tensor.Shape())
			} else {
				newTensors[name] = tensor
			}
		}

		// Write back BF16 safetensor file
		newFile := filepath.Join(bf16Path, filepath.Base(file))
		serialized, err := safetensors.Serialize(newTensors, nil)
		if err != nil {
			return err
		}
		if err := os.WriteFile(newFile, serialized, 0644); err != nil {
			return err
		}
	}

	// Update model index: remove scale_inv entries
	for _, name := range fp8WeightNames {
		delete(weightMap, name+"_scale_inv")
	}
	newIndex := struct {
		Metadata  map[string]interface{} `json:"metadata"`
		WeightMap map[string]string      `json:"weight_map"`
	}{Metadata: map[string]interface{}{}, WeightMap: weightMap}

	outData, err := json.MarshalIndent(newIndex, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(bf16Path, "model.safetensors.index.json"), outData, 0644)
}

// Convert raw bytes of a 2D tensor to gotorch.Tensor (float32)
func bytesToTorchTensor(data []byte, M, N int64) torch.Tensor {
	floats := bytesToFloat32Slice(data)
	return torch.NewTensor(floats).View(M, N)
}

// Convert raw bytes to []float32
func bytesToFloat32Slice(data []byte) []float32 {
	floats := make([]float32, len(data)/4)
	for i := range floats {
		bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
		floats[i] = math.Float32frombits(bits)
	}
	return floats
}

// Convert gotorch.Tensor to safetensors.TensorView
func tensorFromTorch(t torch.Tensor, shape []uint64) safetensors.TensorView {
	flat := torch.Flatten(t, 0, -1)
	numel := int64(1)
	for _, dim := range t.Shape() {
		numel *= dim
	}

	data := make([]byte, numel*4)
	for i := int64(0); i < numel; i++ {
		val := flat.Index(i).Item()
		var f float32
		switch v := val.(type) {
		case float32:
			f = v
		case float64:
			f = float32(v)
		case int:
			f = float32(v)
		case int32:
			f = float32(v)
		case int64:
			f = float32(v)
		default:
			panic(fmt.Sprintf("unsupported tensor dtype: %T", v))
		}
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
	}

	tv, err := safetensors.NewTensorView(safetensors.F32, shape, data)
	if err != nil {
		panic(err)
	}
	return tv
}
