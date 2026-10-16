package llm

import (
	"context"
	"net/http"
	"strings"

	"cyberstrike-ai/internal/config"

	agenticclaude "github.com/cloudwego/eino-ext/components/model/agenticclaude"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

func IsClaudeProvider(provider string) bool {
	provider = strings.ToLower(strings.TrimSpace(provider))
	return provider == "claude" || provider == "anthropic"
}

func NewClaudeAgenticModel(
	ctx context.Context,
	cfg config.OpenAIConfig,
	httpClient *http.Client,
	maxTokens int,
	extraFields map[string]any,
) (model.AgenticModel, error) {
	if maxTokens <= 0 {
		maxTokens = cfg.MaxCompletionTokensEffective()
	}
	if cfg.IsDeepSeekEndpointOrModel() {
		httpClient = newDeepSeekAnthropicCompatibleClient(httpClient)
	}
	httpClient = applyExtraHeadersHTTPClient(httpClient, cfg.ExtraHeaders)
	return agenticclaude.New(ctx, &agenticclaude.Config{
		APIKey:      strings.TrimSpace(cfg.APIKey),
		BaseURL:     strings.TrimSuffix(strings.TrimSpace(cfg.BaseURL), "/"),
		Model:       strings.TrimSpace(cfg.Model),
		MaxTokens:   maxTokens,
		HTTPClient:  httpClient,
		ExtraFields: extraFields,
	})
}

func applyExtraHeadersHTTPClient(base *http.Client, headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return base
	}
	if base == nil {
		base = http.DefaultClient
	}
	cloned := *base
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	cloned.Transport = &extraHeadersRoundTripper{base: transport, headers: headers}
	return &cloned
}

type extraHeadersRoundTripper struct {
	base    http.RoundTripper
	headers map[string]string
}

func (rt *extraHeadersRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	for k, v := range rt.headers {
		if strings.TrimSpace(k) != "" && strings.TrimSpace(v) != "" {
			req.Header.Set(k, v)
		}
	}
	return rt.base.RoundTrip(req)
}

func AgenticText(msg *schema.AgenticMessage) (content, reasoning string) {
	if msg == nil {
		return "", ""
	}
	var contentParts, reasoningParts []string
	for _, block := range msg.ContentBlocks {
		if block == nil {
			continue
		}
		switch {
		case block.AssistantGenText != nil:
			contentParts = append(contentParts, block.AssistantGenText.Text)
		case block.Reasoning != nil:
			reasoningParts = append(reasoningParts, block.Reasoning.Text)
		}
	}
	return strings.Join(contentParts, ""), strings.Join(reasoningParts, "")
}
