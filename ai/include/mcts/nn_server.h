#pragma once
#include "mcts/mcts.h"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// KataGo's NNEvaluator queue (cpp/neuralnet/nneval.cpp): a search builds its position's input
// itself (Evaluator::prepare(), KataGo's fillRow), submits it and blocks; server threads take
// whatever is queued - up to max_batch, never waiting for a full batch (KataGo's waitPopUpToN()) -
// stack it, run it as one batch and split the outputs into rows. Batches form from however many
// searches are waiting at once, and the network runs while other searches do their CPU work.
//
// Server thread i runs evaluators[i % evaluators.size()] (see set_evaluators()) - e.g. one evaluator
// per GPU, each with its own copy of the model there, as KataGo assigns its server threads to GPUs
// (cudaDeviceToUseThread<N>). Threads sharing an evaluator run its model concurrently.
class NNServer {
public:
    NNServer(int num_threads, int max_batch);
    ~NNServer();
    NNServer(const NNServer&) = delete;
    NNServer& operator=(const NNServer&) = delete;

    // Only while no request is pending (e.g. while every search is paused). Server thread i runs
    // evaluators[i % evaluators.size()]. Every evaluator must build the same inputs - searches use
    // the first one's prepare().
    void set_evaluators(std::vector<Evaluator> evaluators);

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
        ModelInput input;
        NNRow row;
        bool done = false;
        std::mutex mutex;
        std::condition_variable cv;
    };

    void serve(int thread_index);

    int max_batch_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Request*> queue_;
    bool closing_ = false;
    std::vector<Evaluator> evaluators_;
    Stats stats_;
    std::vector<std::thread> threads_;
};
