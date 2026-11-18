#include "mcts/nn_server.h"
#include <chrono>

NNServer::NNServer(int num_threads, int max_batch) : max_batch_(max_batch) {
    for (int i = 0; i < num_threads; i++) threads_.emplace_back([this] { serve(); });
}

NNServer::~NNServer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void NNServer::set_evaluator(Evaluator evaluator) {
    std::lock_guard<std::mutex> lock(mutex_);
    evaluator_ = std::move(evaluator);
}

NNRow NNServer::evaluate(const BoardState& state) {
    Request req;
    req.state = &state;
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

void NNServer::serve() {
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
            evaluator = evaluator_;
        }
        std::vector<const BoardState*> states;
        states.reserve(batch.size());
        for (Request* r : batch) states.push_back(r->state);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto rows = evaluate_rows(evaluator, states);
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
