#include "model/transformer.h"
#include <limits>
#include <algorithm>
#include <cassert>
#include <omp.h>

TransformerPolicyHeadImpl::TransformerPolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor TransformerPolicyHeadImpl::forward(const torch::Tensor& h) {
    int64_t B = h.size(0), N = h.size(1);
    auto out          = proj->forward(h);                        // (B, N, ns+1)
    // Node-major (B,N,ns) -> stone-major (B,ns,N) -> flat (B,ns*N), matching legal_mask's
    // stone-major flatten order (same structure as CNN/GNN/UNet's policy heads).
    auto place_logits = out.slice(-1, 0, num_stones_)
                            .permute({0, 2, 1})
                            .reshape({B, num_stones_ * N});        // (B, ns*N)
    auto pass_field   = out.select(-1, num_stones_);               // (B, N)
    auto attn_weights = torch::softmax(attn->forward(h).squeeze(-1), -1); // (B, N)
    auto pass_logit   = (attn_weights * pass_field).sum(-1, /*keepdim=*/true); // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);            // (B, ns*N+1)
}

TransformerBlockImpl::TransformerBlockImpl(int hidden_dim, int num_heads) {
    mha = register_module("mha", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(hidden_dim, num_heads)));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
    ffn = register_module("ffn", torch::nn::Sequential(
        torch::nn::Linear(hidden_dim, 4 * hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(4 * hidden_dim, hidden_dim)));
}

torch::Tensor TransformerBlockImpl::forward(const torch::Tensor& q, const torch::Tensor& kv,
                                             const torch::Tensor& key_padding_mask) {
    auto qn = ln1->forward(q);
    auto kvn = ln1->forward(kv);  // same LN weights applied independently - identical to qn when kv is q (self-attention)
    auto attn_out = std::get<0>(mha->forward(qn, kvn, kvn, key_padding_mask));
    auto x = q + attn_out;
    x = x + ffn->forward(ln2->forward(x));
    return x;
}

TransformerImpl::TransformerImpl(const BoardConfig& bc, const TransformerConfig& cfg,
                                  int num_players, int num_stones)
    : cfg_(cfg), num_players_(num_players), num_stones_(num_stones), N_(bc.N),
      history_feature_dim_(cfg_.history_descr.at("totalDims").get<int>())
{
    assert(cfg_.hidden_dim % kNumHeads == 0 && "TransformerImpl: hidden_dim must be divisible by kNumHeads");

    int D = cfg_.hidden_dim;
    encoder_in  = register_module("encoder_in",  torch::nn::Linear(N_ * cfg_.feature_dim, D));
    encoder_out = register_module("encoder_out", torch::nn::Linear(D, D));

    hist_encoder_in  = register_module("hist_encoder_in",  torch::nn::Linear(N_ * history_feature_dim_, D));
    hist_encoder_out = register_module("hist_encoder_out", torch::nn::Linear(D, D));

    for (int i = 0; i < cfg_.num_attn_layers; i++)
        self_attn_layers_.push_back(register_module("self_attn_" + std::to_string(i), TransformerBlock(D, kNumHeads)));
    for (int i = 0; i < cfg_.num_attn_layers; i++)
        cross_attn_layers_.push_back(register_module("cross_attn_" + std::to_string(i), TransformerBlock(D, kNumHeads)));

    history_sentinel_ = register_parameter("history_sentinel", torch::randn({1, 1, D}));

    decoder_in  = register_module("decoder_in",  torch::nn::Linear(D, D));
    decoder_out = register_module("decoder_out", torch::nn::Linear(D, N_ * D));

    stone_head = register_module("stone_head", torch::nn::Sequential(
        torch::nn::Linear(D, D),
        torch::nn::ReLU(),
        torch::nn::Linear(D, num_stones + 1)));
    territory_head = register_module("territory_head", torch::nn::Sequential(
        torch::nn::Linear(D, D),
        torch::nn::ReLU(),
        torch::nn::Linear(D, num_stones + 1)));
    policy_head = register_module("policy_head", TransformerPolicyHead(D, num_stones));
}

std::pair<torch::Tensor, torch::Tensor> TransformerImpl::forward(
    torch::Tensor hist_x, torch::Tensor hist_mask,
    torch::Tensor cur_x, torch::Tensor legal_mask)
{
    bool batched = (cur_x.dim() == 3);
    if (!batched) {
        hist_x     = hist_x.unsqueeze(0);      // (1,T,N,F)
        hist_mask  = hist_mask.unsqueeze(0);   // (1,T)
        cur_x      = cur_x.unsqueeze(0);       // (1,N,F)
        legal_mask = legal_mask.unsqueeze(0);
    }

    int64_t B = cur_x.size(0);
    int64_t T = hist_x.size(1);
    int D = cfg_.hidden_dim;

    // Encode the current state.
    auto cur_emb = torch::relu(encoder_in->forward(cur_x.reshape({B, N_ * cfg_.feature_dim})));
    cur_emb = encoder_out->forward(cur_emb); // (B, D)

    // Encode every past state with the SAME shared weights (a separate encoder from the current
    // state's, since past plies use the narrower history_descr, a different width).
    torch::Tensor hist_emb;
    if (T > 0) {
        auto hist_flat = hist_x.reshape({B * T, N_ * history_feature_dim_});
        auto he = torch::relu(hist_encoder_in->forward(hist_flat));
        he = hist_encoder_out->forward(he);
        hist_emb = he.reshape({B, T, D});
    } else {
        hist_emb = torch::zeros({B, 0, D}, cur_x.options());
    }

    // Prepend the never-masked history sentinel, so attention always has >=1 valid key even
    // when T==0 (a genesis state has no past plies at all).
    auto sentinel = history_sentinel_.expand({B, 1, D});
    auto seq = torch::cat({sentinel, hist_emb}, 1);                        // (B, T+1, D)
    auto sentinel_mask = torch::zeros({B, 1}, hist_mask.options());
    auto full_mask = torch::cat({sentinel_mask, hist_mask}, 1);            // (B, T+1) True=pad

    // History self-attention - permutation-EQUIVARIANT (no positional/recency signal is ever
    // added to seq), so the history set stays order-symmetric by construction.
    auto seq_lbd = seq.transpose(0, 1);  // (T+1, B, D) - MultiheadAttention needs seq-first layout
    for (auto& blk : self_attn_layers_)
        seq_lbd = blk->forward(seq_lbd, seq_lbd, full_mask);

    // Cross-attention: the current state's embedding is the query, never joining the history set
    // itself - a structurally different role, not just a learned tag.
    auto q = cur_emb.unsqueeze(0);  // (1, B, D)
    for (auto& blk : cross_attn_layers_)
        q = blk->forward(q, seq_lbd, full_mask);
    auto h_star = q.squeeze(0);  // (B, D)

    // Decode back to a per-node representation - the literal inverse of the encoder's flatten,
    // no per-node operation anywhere in this path.
    auto h_nodes = torch::relu(decoder_in->forward(h_star));
    h_nodes = decoder_out->forward(h_nodes).reshape({B, N_, D});

    auto stone_est     = torch::softmax(stone_head->forward(h_nodes), -1);      // (B, N, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h_nodes), -1);  // (B, N, num_stones+1)
    auto ownership = torch::stack({stone_est, territory_est}, 1);              // (B, 2, N, num_stones+1)

    auto logits = policy_head->forward(h_nodes); // (B, num_stones*N+1)
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    logits = logits.masked_fill(legal_mask.logical_not(), NEG_INF);
    logits = logits.masked_fill(legal_mask.any(-1, true).logical_not(), 0.0f);
    auto policy = torch::softmax(logits, -1);

    if (!batched)
        return {policy.squeeze(0), ownership.squeeze(0)};
    return {policy, ownership};
}

std::pair<torch::Tensor, torch::Tensor> TransformerImpl::evaluate(const BoardState& state) {
    torch::NoGradGuard ng;
    auto dev = encoder_in->weight.device();

    int T = state.ply_count();  // number of PAST plies (0..T-1); T itself is the current ply
    std::vector<torch::Tensor> hist_rows(T);
    for (int k = 0; k < T; k++) hist_rows[k] = history_features_at_ply(state, k, cfg_.history_descr);
    auto hist_x = (T > 0) ? torch::stack(hist_rows, 0) : torch::zeros({0, N_, history_feature_dim_}, torch::kFloat32);
    auto hist_mask = torch::zeros({T}, torch::kBool);  // no padding for a single, non-batched state

    auto [cur_x, legal_mask] = board_to_features(state, dev, cfg_.input_descr);
    auto [policy, ownership] = forward(hist_x.to(dev), hist_mask.to(dev), cur_x, legal_mask);
    return {policy, ownership};
}

static std::pair<torch::Tensor, torch::Tensor> run_batch(
    TransformerImpl* self,
    const std::vector<const BoardState*>& states)
{
    torch::NoGradGuard ng;
    auto dev = self->encoder_in->weight.device();
    int B = static_cast<int>(states.size());
    int N = self->N_;
    int Fh = self->history_feature_dim_;

    std::vector<torch::Tensor> cur_feats(B), legal_masks(B), hist_stacks(B);
    std::vector<int> T(B);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B; i++) {
        int Ti = states[i]->ply_count();
        T[i] = Ti;
        std::vector<torch::Tensor> rows(Ti);
        for (int k = 0; k < Ti; k++) rows[k] = history_features_at_ply(*states[i], k, self->cfg_.history_descr);
        hist_stacks[i] = (Ti > 0) ? torch::stack(rows, 0) : torch::zeros({0, N, Fh}, torch::kFloat32);

        auto [ft, mask] = board_to_features(*states[i], dev, self->cfg_.input_descr);
        cur_feats[i] = ft;
        legal_masks[i] = mask;
    }

    int Tmax = 0;
    for (int t : T) Tmax = std::max(Tmax, t);

    auto hist_x    = torch::zeros({B, Tmax, N, Fh}, torch::kFloat32);
    auto hist_mask = torch::ones({B, Tmax}, torch::kBool);  // True = pad
    for (int i = 0; i < B; i++) {
        if (T[i] > 0) {
            hist_x[i].slice(0, 0, T[i]).copy_(hist_stacks[i]);
            hist_mask[i].slice(0, 0, T[i]).fill_(false);
        }
    }
    auto cur_x    = torch::stack(cur_feats, 0);   // (B, N, F) - already on dev
    auto legal_mask = torch::stack(legal_masks, 0); // (B, ns*N+1) - already on dev

    return self->forward(hist_x.to(dev), hist_mask.to(dev), cur_x, legal_mask);
}

std::pair<torch::Tensor, torch::Tensor> TransformerImpl::evaluate_batch(
    const std::vector<BoardState*>& states)
{
    std::vector<const BoardState*> cstates(states.begin(), states.end());
    return run_batch(this, cstates);
}

std::pair<torch::Tensor, torch::Tensor> TransformerImpl::evaluate_batch(
    const std::vector<const BoardState*>& states)
{
    return run_batch(this, states);
}
