#pragma once
#include <vector>
#include <queue>
#include "event.hpp"

// Mirror of backtesting/data_handler.py.
//
// Holds the full price matrix in memory as a flat, column-major-by-ticker
// layout: prices_[ticker] is a contiguous vector of that ticker's adj_close
// over time. Walking one ticker forward in time is therefore a sequential
// memory scan (cache-friendly) -- the opposite of Python's pandas .iterrows()
// which built a fresh dict per row per bar.
class DataHandler {
public:
    DataHandler(std::vector<std::vector<double>> prices)
        : prices_(std::move(prices)),
          n_tickers_(static_cast<int>(prices_.size())),
          n_bars_(prices_.empty() ? 0 : static_cast<int>(prices_[0].size())),
          current_index_(-1),
          continue_backtest_(true) {}

    // Advance to the next bar and push exactly ONE MarketEvent, mirroring the
    // Python generator that emits a single (empty) MarketEvent per time step
    // regardless of how many tickers exist.
    void update_bars(std::queue<Event>& events) {
        if (current_index_ + 1 < n_bars_) {
            ++current_index_;
            events.push(MarketEvent{});
        } else {
            continue_backtest_ = false;  // StopIteration equivalent
        }
    }

    bool   continue_backtest() const { return continue_backtest_; }
    int    n_tickers()          const { return n_tickers_; }
    int    n_bars()             const { return n_bars_; }
    long   timeindex()          const { return current_index_; }

    // Latest observed adj_close for a ticker (0.0 before the first bar).
    double price(int ticker) const {
        if (current_index_ < 0) return 0.0;
        return prices_[ticker][current_index_];
    }

    // Historical price for the SMA windows (bar may be negative -> not ready).
    double price_at(int ticker, int bar) const {
        if (bar < 0 || bar >= n_bars_) return 0.0;
        return prices_[ticker][bar];
    }

private:
    std::vector<std::vector<double>> prices_;
    int  n_tickers_;
    int  n_bars_;
    int  current_index_;
    bool continue_backtest_;
};
