#pragma once
#include <queue>
#include <utility>
#include "event.hpp"
#include "data_handler.hpp"
#include "strategy.hpp"
#include "portfolio.hpp"
#include "execution.hpp"

// Mirror of backtesting/engine.py::BacktestingEngine.
//
// The invariant that must survive the port: for each bar we push exactly one
// MarketEvent, then DRAIN the queue completely before advancing time. A
// MarketEvent can cascade Signal -> Order -> Fill within the same timestamp;
// only once no events remain do we move to the next bar. That ordering is what
// makes look-ahead bias structurally impossible.
class BacktestingEngine {
public:
    explicit BacktestingEngine(DataHandler data, double initial_capital = 100000.0)
        : data_(std::move(data)),
          initial_capital_(initial_capital),
          strategy_(data_),
          portfolio_(data_, initial_capital),
          execution_(data_) {}

    struct Result {
        double final_value;
        double total_return_pct;
        long   signals;
        long   orders;
        long   fills;
        long   bars;
    };

    Result run() {
        signals_ = orders_ = fills_ = 0;

        while (true) {
            if (data_.continue_backtest()) {
                data_.update_bars(events_);
            } else {
                break;
            }
            // Inner loop: drain every event this bar produced.
            while (!events_.empty()) {
                Event e = std::move(events_.front());
                events_.pop();
                std::visit(overloaded{
                    [&](MarketEvent&) {
                        strategy_.calculate_signals(events_);
                        portfolio_.update_timeindex();
                    },
                    [&](SignalEvent& s) { ++signals_; portfolio_.update_signal(s, events_); },
                    [&](OrderEvent&  o) { ++orders_;  execution_.execute_order(o, events_); },
                    [&](FillEvent&   f) { ++fills_;   portfolio_.update_fill(f); }
                }, e);
            }
        }

        const double final_value = portfolio_.total_value();
        Result r;
        r.final_value      = final_value;
        r.total_return_pct = (final_value - initial_capital_) / initial_capital_ * 100.0;
        r.signals = signals_;
        r.orders  = orders_;
        r.fills   = fills_;
        r.bars    = data_.n_bars();
        return r;
    }

    const Portfolio& portfolio() const { return portfolio_; }

private:
    DataHandler        data_;
    double             initial_capital_;
    std::queue<Event>  events_;      // deque-backed FIFO; reused across bars
    Strategy           strategy_;
    Portfolio          portfolio_;
    ExecutionHandler   execution_;
    long signals_ = 0, orders_ = 0, fills_ = 0;
};
