#include "model/heads.h"
#include <limits>

PolicyHeadImpl::PolicyHeadImpl(int hidden_dim, int num_stones)
    : num_stones_(num_stones)
{
    proj = register_module("proj", torch::nn::Linear(hidden_dim, num_stones + 1));
    attn = register_module("attn", torch::nn::Linear(hidden_dim, 1));
}

torch::Tensor PolicyHeadImpl::forward(const torch::Tensor& h) {
    int64_t B = h.size(0), S = h.size(1);
    auto out          = proj->forward(h);                        // (B, S, ns+1)
    // Place logits are slot-major out of `proj` ((B,S,numStones)) but must be flattened
    // stone-major (index (stone-1)*S+slot, matching legal_mask's layout - see slot_input(),
    // features.h) - permute before reshaping to make this happen.
    auto place_logits = out.slice(-1, 0, num_stones_)
                            .permute({0, 2, 1})
                            .reshape({B, num_stones_ * S});        // (B, ns*S)
    auto pass_field   = out.select(-1, num_stones_);               // (B, S)
    auto attn_weights = torch::softmax(attn->forward(h).squeeze(-1), -1); // (B, S)
    auto pass_logit   = (attn_weights * pass_field).sum(-1, /*keepdim=*/true); // (B, 1)
    return torch::cat({place_logits, pass_logit}, -1);            // (B, ns*S+1)
}

RankHeadImpl::RankHeadImpl(int hidden_dim, int num_pairs) : num_pairs_(num_pairs) {
    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(2 * hidden_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, num_pairs * 3)));
}

torch::Tensor RankHeadImpl::forward(const torch::Tensor& h) {
    auto pooled = torch::cat({h.mean(1), std::get<0>(h.max(1))}, -1);   // (B, 2*hidden_dim)
    auto logits = mlp->forward(pooled).reshape({h.size(0), num_pairs_, 3});
    return torch::softmax(logits, -1);
}

static torch::nn::Sequential ownership_head(int hidden_dim, int num_stones) {
    return torch::nn::Sequential(
        torch::nn::Linear(hidden_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, num_stones + 1));
}

void OutputHeads::register_on(torch::nn::Module& owner, int hidden_dim, int num_stones, int num_players) {
    stone_head     = owner.register_module("stone_head", ownership_head(hidden_dim, num_stones));
    territory_head = owner.register_module("territory_head", ownership_head(hidden_dim, num_stones));
    policy_head    = owner.register_module("policy_head", PolicyHead(hidden_dim, num_stones));
    rank_head      = owner.register_module("rank_head", RankHead(hidden_dim, num_rank_pairs(num_players)));
}

ModelOutput OutputHeads::forward(const torch::Tensor& h, const torch::Tensor& legal_mask) {
    auto stone_est     = torch::softmax(stone_head->forward(h), -1);      // (B, S, num_stones+1)
    auto territory_est = torch::softmax(territory_head->forward(h), -1);  // (B, S, num_stones+1)
    auto ownership     = torch::stack({stone_est, territory_est}, 1);     // (B, 2, S, num_stones+1)
    auto rank          = rank_head->forward(h);                           // (B, num_rank_pairs, 3)

    auto logits = policy_head->forward(h);                                // (B, num_stones*S+1)
    logits = logits.masked_fill(legal_mask.logical_not(), -std::numeric_limits<float>::infinity());
    // An all-illegal row would otherwise softmax to NaN.
    logits = logits.masked_fill(legal_mask.any(-1, true).logical_not(), 0.0f);
    return {torch::softmax(logits, -1), ownership, rank};
}
