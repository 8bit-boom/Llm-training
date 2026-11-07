// Renders an Ollama-style Go template (text/template) for a list of messages, the way Ollama does for a chat request: .Messages with .Role and
// .Content. Used by the Rust tests to check the generated Modelfile templates against the prompts the models were trained on.
// Usage: go run main.go <template file> <messages json file>
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"text/template"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Values struct {
	Messages []Message
	System   string
	Prompt   string
	Response string
}

func main() {
	tpl, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	raw, err := os.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var msgs []Message
	if err := json.Unmarshal(raw, &msgs); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	t, err := template.New("m").Parse(string(tpl))
	if err != nil {
		fmt.Fprintln(os.Stderr, "parse:", err)
		os.Exit(1)
	}
	if err := t.Execute(os.Stdout, Values{Messages: msgs}); err != nil {
		fmt.Fprintln(os.Stderr, "execute:", err)
		os.Exit(1)
	}
}
