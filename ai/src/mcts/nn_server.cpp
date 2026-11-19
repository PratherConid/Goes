#include "mcts/nn_server.h"
#include <chrono>
#include <stdexcept>

NNServer::NNServer(int num_threads, int max_batch) : max_batch_(max_batch) {
    for (int i = 0; i < num_threads; i++) threads_.emplace_back([this, i] { serve(i); });
}

NNServer::~NNServer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void NNServer::set_evaluators(std::vector<Evaluator> evaluators) {
    if (evaluators.empty()) throw std::invalid_argument("NNServer::set_evaluators: no evaluators");
    std::lock_guard<std::mutex> lock(mutex_);
    evaluators_ = std::move(evaluators);
}

NNRow NNServer::evaluate(const BoardState& state) {
    Request req;
    req.state = &state;
    req.input = evaluators_.front().prepare(state);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(&req);
    }
    cv_.notify_one();
    std::unique_lock<std::mutex> lock(req.mutex);
    req.cv.wait(lock, [&] { return req.done; });
    return std::move(req.row);
}

NNServer::Stats NNServer::take_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats s = stats_;
    stats_ = Stats{};
    return s;
}

void NNServer::serve(int thread_index) {
    while (true) {
        std::vector<Request*> batch;
        Evaluator evaluator;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return closing_ || !queue_.empty(); });
            if (queue_.empty()) return;  // closing
            while (!queue_.empty() && (int)batch.size() < max_batch_) {
                batch.push_back(queue_.front());
                queue_.pop_front();
            }
            evaluator = evaluators_[thread_index % evaluators_.size()];
        }
        std::vector<const ModelInput*> inputs;
        std::vector<const BoardState*> states;
        inputs.reserve(batch.size());
        states.reserve(batch.size());
        for (Request* r : batch) {
            inputs.push_back(&r->input);
            states.push_back(r->state);
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        auto rows = output_rows(evaluator.run(inputs), states);
        double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.rows += (int64_t)batch.size();
            stats_.batches += 1;
            stats_.seconds += secs;
        }
        for (size_t i = 0; i < batch.size(); i++) {
            Request* r = batch[i];
            std::lock_guard<std::mutex> lock(r->mutex);
            r->row = std::move(rows[i]);
            r->done = true;
            r->cv.notify_one();
        }
    }
}
