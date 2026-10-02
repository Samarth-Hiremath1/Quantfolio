#pragma once
#include <vector>
#include <queue>
#include "event.hpp"
#include "data_handler.hpp"

// Mirror of backtesting/portfolio.py.
//  - fixed 100-share sizing
//  - tracks cash and per-ticker share counts
//  - records an equity-curve point (total value) once per bar
class Portfolio {
public:
    Portfolio(const DataHandler& data, double initial_capital = 100000.0)
        : data_(data),
          cash_(initial_capital),
          positions_(data.n_tickers(), 0) {
        equity_.reserve(static_cast<size_t>(data.n_bars()));
    }

    // On a MARKET event: mark holdings to market and record equity.
    void update_timeindex() {
        double holdings = 0.0;
        for (int t = 0; t < data_.n_tickers(); ++t)
            holdings += positions_[t] * data_.price(t);
        equity_.push_back(cash_ + holdings);
    }

    // On a SIGNAL event: turn it into a fixed-lot market order.
    void update_signal(const SignalEvent& s, std::queue<Event>& events) {
        OrderEvent o;
        o.ticker    = s.ticker;
        o.quantity  = 100;  // fixed lot, mirrors Python
        o.direction = (s.signal_type == SignalType::Long) ? Direction::Buy
                                                          : Direction::Sell;
        events.push(o);
    }

    // On a FILL event: update cash and positions.
    void update_fill(const FillEvent& f) {
        const int dir = (f.direction == Direction::Buy) ? 1 : -1;
        positions_[f.ticker] += dir * f.quantity;
        const double cost = dir * f.fill_cost * f.quantity;
        cash_ -= (cost + f.commission);
    }

    double cash() const { return cash_; }

    double total_value() const {
        double holdings = 0.0;
        for (int t = 0; t < data_.n_tickers(); ++t)
            holdings += positions_[t] * data_.price(t);
        return cash_ + holdings;
    }

    const std::vector<double>& equity_curve() const { return equity_; }

private:
    const DataHandler&  data_;
    double              cash_;
    std::vector<int>    positions_;
    std::vector<double> equity_;
};
