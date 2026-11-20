#include "model/transformer.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

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
    auto kvn = ln1->forward(kv);  // same LN weights as qn, applied independently to the key/value sequence
    auto attn_out = std::get<0>(mha->forward(qn, kvn, kvn, key_padding_mask));
    auto x = q + attn_out;
    x = x + ffn->forward(ln2->forward(x));
    return x;
}

TransformerImpl::TransformerImpl(const TransformerConfig& cfg, int num_stones, int num_players)
    : cfg_(cfg), num_slots_(num_node_slots(cfg.input_descr)),
      history_feature_dim_(cfg_.history_descr.at("totalDims").get<int>())
{
    if (cfg_.hidden_dim % kNumHeads != 0)
        throw std::runtime_error("Transformer: hidden_dim " + std::to_string(cfg_.hidden_dim) +
                                 " must be divisible by its " + std::to_string(kNumHeads) + " attention heads");

    int D = cfg_.hidden_dim;
    encoder_in  = register_module("encoder_in",  torch::nn::Linear(num_slots_ * cfg_.feature_dim, D));
    encoder_out = register_module("encoder_out", torch::nn::Linear(D, D));

    hist_encoder_in  = register_module("hist_encoder_in",  torch::nn::Linear(num_slots_ * history_feature_dim_, D));
    hist_encoder_out = register_module("hist_encoder_out", torch::nn::Linear(D, D));

    for (int i = 0; i < cfg_.num_attn_layers; i++)
        cross_attn_layers_.push_back(register_module("cross_attn_" + std::to_string(i), TransformerBlock(D, kNumHeads)));

    history_sentinel_ = register_parameter("history_sentinel", torch::randn({1, 1, D}));

    decoder_in  = register_module("decoder_in",  torch::nn::Linear(D, D));
    decoder_out = register_module("decoder_out", torch::nn::Linear(D, num_slots_ * D));

    heads.register_on(*this, D, num_stones, num_players);
}

ModelOutput TransformerImpl::forward(
    torch::Tensor hist_x, torch::Tensor hist_mask,
    torch::Tensor cur_x, torch::Tensor legal_mask)
{
    int64_t B = cur_x.size(0);
    int64_t T = hist_x.size(1);
    int D = cfg_.hidden_dim;

    // Encode the current state.
    auto cur_emb = torch::relu(encoder_in->forward(cur_x.reshape({B, num_slots_ * cfg_.feature_dim})));
    cur_emb = encoder_out->forward(cur_emb); // (B, D)

    // Encode every past state with the SAME shared weights (a separate encoder from the current
    // state's, since past plies use the narrower history_descr, a different width).
    torch::Tensor hist_emb;
    if (T > 0) {
        auto hist_flat = hist_x.reshape({B * T, num_slots_ * history_feature_dim_});
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

    // seq is permutation-EQUIVARIANT (no positional/recency signal is ever added), so the history
    // set stays order-symmetric by construction - used as-is below, no self-attention refinement.
    auto seq_lbd = seq.transpose(0, 1);  // (T+1, B, D) - MultiheadAttention needs seq-first layout

    // Cross-attention: the current state's embedding is the query, never joining the history set
    // itself - a structurally different role, not just a learned tag. The key/value set (seq_lbd)
    // stays fixed across layers; only the query evolves.
    auto q = cur_emb.unsqueeze(0);  // (1, B, D)
    for (auto& blk : cross_attn_layers_)
        q = blk->forward(q, seq_lbd, full_mask);
    auto h_star = q.squeeze(0);  // (B, D)

    // Decode back to a per-slot representation - the literal inverse of the encoder's flatten, no
    // per-slot operation anywhere in this path.
    auto h_slots = torch::relu(decoder_in->forward(h_star));
    h_slots = decoder_out->forward(h_slots).reshape({B, num_slots_, D});  // (B, S, D)

    return heads.forward(h_slots, legal_mask);
}

ModelInput TransformerImpl::prepare_input(const BoardState& state, const std::shared_ptr<const GameInput>& game) const {
    ModelInput in = slot_input(state, cfg_.input_descr, game);
    int T = state.ply_count();
    if (T > 0) {
        std::vector<torch::Tensor> rows(T);
        for (int k = 0; k < T; k++) rows[k] = history_features_at_ply(state, k, cfg_.history_descr);
        in.history = nodes_to_slots(stack_cpu(rows), game->slots, 1);
    } else {
        in.history = torch::empty({0, num_slots_, history_feature_dim_}, torch::kFloat32);
    }
    return in;
}

ModelOutput TransformerImpl::evaluate_inputs(const std::vector<const ModelInput*>& inputs) {
    torch::NoGradGuard ng;
    auto dev = encoder_in->weight.device();
    int64_t B = static_cast<int64_t>(inputs.size());
    int64_t Tmax = 0;
    for (const ModelInput* in : inputs) Tmax = std::max(Tmax, in->history.size(0));

    // Each input's history, zero-padded to Tmax plies, copied in with plain memcpys/memsets for the
    // same reason as stack_cpu() (model_output.h).
    auto hist_x    = torch::empty({B, Tmax, num_slots_, history_feature_dim_}, torch::kFloat32);
    auto hist_mask = torch::empty({B, Tmax}, torch::kBool);  // True = pad
    size_t ply_bytes = static_cast<size_t>(num_slots_) * history_feature_dim_ * sizeof(float);
    auto* hist_dst = static_cast<char*>(hist_x.data_ptr());
    bool* mask_dst = hist_mask.data_ptr<bool>();
    for (int64_t i = 0; i < B; i++) {
        const torch::Tensor& h = inputs[i]->history;
        int64_t T = h.size(0);
        TORCH_CHECK(h.is_contiguous(), "Transformer: history input must be contiguous");
        char* row = hist_dst + static_cast<size_t>(i * Tmax) * ply_bytes;
        if (T > 0) std::memcpy(row, h.data_ptr(), static_cast<size_t>(T) * ply_bytes);
        std::memset(row + static_cast<size_t>(T) * ply_bytes, 0, static_cast<size_t>(Tmax - T) * ply_bytes);
        std::fill(mask_dst + i * Tmax, mask_dst + i * Tmax + T, false);
        std::fill(mask_dst + i * Tmax + T, mask_dst + (i + 1) * Tmax, true);
    }
    auto cur_x      = stack_inputs(inputs, &ModelInput::features).to(dev);   // (B, S, F)
    auto legal_mask = stack_inputs(inputs, &ModelInput::legal_mask).to(dev); // (B, ns*S+1)
    return forward(hist_x.to(dev), hist_mask.to(dev), cur_x, legal_mask);
}
