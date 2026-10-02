#pragma once
#include <vector>
#include <queue>
#include "event.hpp"
#include "data_handler.hpp"

// Mirror of backtesting/strategy.py::MLForecastStrategy in its FIXED form:
// a real SMA(fast/slow) crossover that fires ONLY on a change of state, never
// on every bar, and never from random draws.
//
// SMA is maintained with rolling sums in O(1) per ticker per bar: when a new
// price arrives we add it to both window sums and subtract the price leaving
// each window (read straight out of the DataHandler, so we keep no duplicate
// history). No allocation happens here in steady state.
class Strategy {
public:
    Strategy(const DataHandler& data, int fast_window = 10, int slow_window = 50)
        : data_(data),
          fast_(fast_window),
          slow_(slow_window),
          fast_sum_(data.n_tickers(), 0.0),
          slow_sum_(data.n_tickers(), 0.0),
          state_(data.n_tickers(), 0) {}

    void calculate_signals(std::queue<Event>& events) {
        const int bar = static_cast<int>(data_.timeindex());
        const int n   = data_.n_tickers();
        for (int t = 0; t < n; ++t) {
            const double p = data_.price_at(t, bar);

            // Rolling window update (bar is 0-based; count of prices seen = bar+1).
            fast_sum_[t] += p;
            if (bar - fast_ >= 0) fast_sum_[t] -= data_.price_at(t, bar - fast_);
            slow_sum_[t] += p;
            if (bar - slow_ >= 0) slow_sum_[t] -= data_.price_at(t, bar - slow_);

            // Need a full slow window before deciding (matches Python guard).
            if (bar + 1 < slow_) continue;

            const double fast_ma = fast_sum_[t] / fast_;
            const double slow_ma = slow_sum_[t] / slow_;
            const int desired = (fast_ma > slow_ma) ? 1 : -1;

            if (desired == state_[t]) continue;   // fire only on state change
            state_[t] = desired;

            SignalEvent s;
            s.ticker      = t;
            s.timeindex   = data_.timeindex();
            s.signal_type = (desired == 1) ? SignalType::Long : SignalType::Short;
            s.strength    = 1.0;
            events.push(s);
        }
    }

private:
    const DataHandler&  data_;
    int                 fast_;
    int                 slow_;
    std::vector<double> fast_sum_;
    std::vector<double> slow_sum_;
    std::vector<int>    state_;   // -1 short, 0 none, +1 long
};
