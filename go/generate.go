// Package main of GoSeek-V3 — A Go runtime for DeepSeek models.
//
// This file is a Go rewrite of the original Python implementation
// designed to run DeepSeek models.
// Original by DeepSeek: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/generate.py
//
// Modified by Lennard <leycm@proton.me>, 2026.
//
// This project is not affiliated with or endorsed by DeepSeek.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
)

// sample draws a token from logits using temperature scaling.
// Mirrors Python: probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
// which is equivalent to Gumbel-max / exponential-race sampling.
func sample(logits []float32, temperature float32) int {
	if temperature < 1e-5 {
		temperature = 1e-5
	}

	// scale logits by temperature, then softmax
	maxL := logits[0]
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}

	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxL) / temperature)))
		probs[i] = e
		sum += e
	}
	for i := range probs {
		probs[i] /= sum
	}

	// divide each prob by Exponential(1) sample, take argmax —
	// equivalent to the Python exponential_race trick
	best := 0
	bestVal := float32(-1)
	for i, p := range probs {
		if p == 0 {
			continue
		}
		// Exponential(1) variate: -ln(U), U ~ Uniform(0,1)
		u := rand.Float64()
		if u == 0 {
			u = 1e-38
		}
		race := p / float32(-math.Log(u))
		if race > bestVal {
			bestVal = race
			best = i
		}
	}
	return best
}

// generate produces completion tokens for a batch of prompt token sequences.
// Mirrors the Python generate() function exactly, including the prompt-mask
// logic and early-stop on EOS.
func generate(
	model *Transformer,
	promptTokens [][]int,
	maxNewTokens int,
	eosID int,
	temperature float32,
) [][]int {
	promptLens := make([]int, len(promptTokens))
	maxPrompt := 0
	for i, t := range promptTokens {
		promptLens[i] = len(t)
		if len(t) > maxPrompt {
			maxPrompt = len(t)
		}
	}

	if maxPrompt > model.args.MaxSeqLen {
		panic(fmt.Sprintf(
			"prompt length exceeds model maximum sequence length (max_seq_len=%d)",
			model.args.MaxSeqLen,
		))
	}

	totalLen := model.args.MaxSeqLen
	if tokenCap := maxNewTokens + maxPrompt; tokenCap < totalLen {
		totalLen = tokenCap
	}

	bsz := len(promptTokens)
	// tokens[b][pos] == -1 means "not yet filled"
	tokens := make([][]int, bsz)
	for i := range tokens {
		tokens[i] = make([]int, totalLen)
		for j := range tokens[i] {
			tokens[i][j] = -1
		}
		copy(tokens[i], promptTokens[i])
	}

	finished := make([]bool, bsz)
	prevPos := 0

	for curPos := maxPrompt; curPos < totalLen; curPos++ {
		// build input window [prevPos, curPos)
		window := make([][]int, bsz)
		for b := range tokens {
			window[b] = tokens[b][prevPos:curPos]
		}

		logitsTensor := model.Forward(window, prevPos)
		// logitsTensor is shaped (bsz, vocabSize/worldSize) — flat row-major
		vocabPart := logitsTensor.numel() / bsz

		for b := 0; b < bsz; b++ {
			if finished[b] {
				continue
			}

			rowLogits := logitsTensor.Data[b*vocabPart : (b+1)*vocabPart]

			var nextToken int
			// prompt_mask[:, cur_pos]: if the position is still within the
			// original prompt, keep the ground-truth token.
			if curPos < promptLens[b] {
				nextToken = tokens[b][curPos]
			} else {
				if temperature > 0 {
					nextToken = sample(rowLogits, temperature)
				} else {
					// greedy
					best := 0
					for j, v := range rowLogits {
						if v > rowLogits[best] {
							best = j
						}
					}
					nextToken = best
				}
			}

			tokens[b][curPos] = nextToken

			if curPos >= promptLens[b] && nextToken == eosID {
				finished[b] = true
			}
		}

		prevPos = curPos

		allDone := true
		for _, f := range finished {
			if !f {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}
	}

	// trim to completion tokens only (after prompt, up to maxNewTokens / EOS)
	completions := make([][]int, bsz)
	for b, toks := range tokens {
		start := promptLens[b]
		end := start + maxNewTokens
		if end > len(toks) {
			end = len(toks)
		}
		slice := toks[start:end]
		// cut at first EOS
		for i, t := range slice {
			if t == eosID {
				slice = slice[:i]
				break
			}
		}
		completions[b] = slice
	}
	return completions
}

// chatMessage mirrors the Python dict{"role": ..., "content": ...}.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// applySimpleChatTemplate is a minimal stand-in for
// tokenizer.apply_chat_template. Real deployments should call the
// actual HuggingFace tokenizer via cgo or a subprocess; this stub
// lets the file compile and run end-to-end for testing.
//
// Format mirrors the DeepSeek instruct chat template:
//
//	<|begin_of_sentence|>User: {msg}\n\nAssistant:
func applySimpleChatTemplate(messages []chatMessage) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_sentence|>")
	for _, m := range messages {
		switch m.Role {
		case "user":
			sb.WriteString("User: ")
			sb.WriteString(m.Content)
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString("Assistant: ")
			sb.WriteString(m.Content)
			sb.WriteString("\n\n")
		}
	}
	sb.WriteString("Assistant:")
	return sb.String()
}

// naiveTokenize is a placeholder that encodes each rune as its Unicode
// codepoint. Replace with a real BPE tokenizer for production use.
func naiveTokenize(text string) []int {
	runes := []rune(text)
	ids := make([]int, len(runes))
	for i, r := range runes {
		ids[i] = int(r)
	}
	return ids
}

// naiveDetokenize is the inverse of naiveTokenize.
func naiveDetokenize(ids []int) string {
	runes := make([]rune, len(ids))
	for i, id := range ids {
		runes[i] = rune(id)
	}
	return string(runes)
}

// mainGenerate is the entry-point called by main() and mirrors the Python
// main() function in generate.py with identical CLI flags.
func mainGenerate() {
	ckptPath := flag.String("ckpt-path", "", "Path to the model checkpoint directory")
	configPath := flag.String("config", "", "Path to the model configuration file")
	inputFile := flag.String("input-file", "", "File containing prompts for batch processing")
	interactive := flag.Bool("interactive", false, "Enable interactive mode")
	maxNewTokens := flag.Int("max-new-tokens", 200, "Maximum number of new tokens to generate")
	temperature := flag.Float64("temperature", 0.2, "Temperature for sampling")
	flag.Parse()

	if *ckptPath == "" || *configPath == "" {
		_, err := fmt.Fprintln(os.Stderr, "both --ckpt-path and --config are required")
		if err != nil {
			return
		}
		os.Exit(1)
	}
	if !*interactive && *inputFile == "" {
		_, err := fmt.Fprintln(os.Stderr, "either --input-file or --interactive must be specified")
		if err != nil {
			return
		}
		os.Exit(1)
	}

	// load config
	cfgData, err := os.ReadFile(*configPath)
	if err != nil {
		_, err := fmt.Fprintln(os.Stderr, "cannot read config:", err)
		if err != nil {
			return
		}
		os.Exit(1)
	}
	args := DefaultModelArgs()
	if err := json.Unmarshal(cfgData, &args); err != nil {
		_, err := fmt.Fprintln(os.Stderr, "cannot parse config:", err)
		if err != nil {
			return
		}
		os.Exit(1)
	}
	fmt.Println(args)

	// build model
	model := NewTransformer(args)

	// warm-up decode (mirrors Python: tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0]))
	warmup := naiveTokenize("DeepSeek")
	warmupOut := generate(model, [][]int{warmup}, 2, -1, 1.0)
	_ = naiveDetokenize(warmupOut[0])

	// load weights — placeholder; real impl uses safetensors loader
	// load_model(model, filepath.Join(*ckptPath, fmt.Sprintf("model%d-mp%d.safetensors", rank, worldSize)))
	_ = *ckptPath // suppress unused warning until real loader is wired in

	eosID := 0 // replace with tokenizer.eos_token_id

	temp32 := float32(*temperature)

	if *interactive {
		var messages []chatMessage
		scanner := bufio.NewScanner(os.Stdin)

		for {
			fmt.Print(">>> ")
			if !scanner.Scan() {
				break
			}
			prompt := scanner.Text()

			switch prompt {
			case "/exit":
				return
			case "/clear":
				messages = messages[:0]
				continue
			}

			messages = append(messages, chatMessage{Role: "user", Content: prompt})

			tmpl := applySimpleChatTemplate(messages)
			promptToks := naiveTokenize(tmpl)

			completionToks := generate(model, [][]int{promptToks}, *maxNewTokens, eosID, temp32)
			completion := naiveDetokenize(completionToks[0])

			fmt.Println(completion)
			messages = append(messages, chatMessage{Role: "assistant", Content: completion})
		}
		return
	}

	// batch mode
	data, err := os.ReadFile(*inputFile)
	if err != nil {
		_, err := fmt.Fprintln(os.Stderr, "cannot read input file:", err)
		if err != nil {
			return
		}
		os.Exit(1)
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	prompts := make([]string, 0, len(lines))
	for _, l := range lines {
		if t := strings.TrimSpace(l); t != "" {
			prompts = append(prompts, t)
		}
	}

	if len(prompts) > args.MaxBatchSize {
		_, err := fmt.Fprintf(os.Stderr,
			"number of prompts (%d) exceeds maximum batch size (%d)\n",
			len(prompts), args.MaxBatchSize,
		)
		if err != nil {
			return
		}
		os.Exit(1)
	}

	promptTokens := make([][]int, len(prompts))
	for i, p := range prompts {
		msg := []chatMessage{{Role: "user", Content: p}}
		promptTokens[i] = naiveTokenize(applySimpleChatTemplate(msg))
	}

	completionTokens := generate(model, promptTokens, *maxNewTokens, eosID, temp32)

	for i, toks := range completionTokens {
		fmt.Println("Prompt:", prompts[i])
		fmt.Println("Completion:", naiveDetokenize(toks))
		fmt.Println()
	}
}

func main() {
	mainGenerate()
}
