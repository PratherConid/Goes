#pragma once
#include "mcts/mcts.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// KataGo's NNEvaluator queue (cpp/neuralnet/nneval.cpp): searches submit one position each and
// block; server threads take whatever is queued - up to max_batch, never waiting for a full batch
// (KataGo's waitPopUpToN()) - and evaluate it as one batch. Batches form from however many searches
// are waiting at once, and the network runs while other searches do their CPU work.
//
// Several server threads share one evaluator, so they run the model concurrently; one thread
// (KataGo's self-play default) avoids that.
class NNServer {
public:
    NNServer(int num_threads, int max_batch);
    ~NNServer();
    NNServer(const NNServer&) = delete;
    NNServer& operator=(const NNServer&) = delete;

    // Only while no request is pending (e.g. while every search is paused).
    void set_evaluator(Evaluator evaluator);

    // Blocks until `state`'s outputs are ready; `state` must stay alive until then.
    NNRow evaluate(const BoardState& state);

    struct Stats {
        int64_t rows = 0;
        int64_t batches = 0;
        double seconds = 0.0;  // time spent evaluating batches
    };
    // Counters since the previous call.
    Stats take_stats();

private:
    struct Request {
        const BoardState* state;
        NNRow row;
        bool done = false;
        std::mutex mutex;
        std::condition_variable cv;
    };

    void serve();

    int max_batch_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request*> queue_;
    bool closing_ = false;
    Evaluator evaluator_;
    Stats stats_;
    std::vector<std::thread> threads_;
};
