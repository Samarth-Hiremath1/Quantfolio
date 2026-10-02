#pragma once
#include <algorithm>
#include <queue>
#include "event.hpp"
#include "data_handler.hpp"

// Mirror of backtesting/execution.py.
//  - 0.05% slippage (buy fills higher, sell fills lower)
//  - Interactive-Brokers-style commission: $0.005/share, $1.00 minimum
//  - instant fill at the current bar's price (+/- slippage)
class ExecutionHandler {
public:
    explicit ExecutionHandler(const DataHandler& data, double slippage_pct = 0.0005)
        : data_(data), slippage_pct_(slippage_pct) {}

    void execute_order(const OrderEvent& o, std::queue<Event>& events) {
        const double price = data_.price(o.ticker);
        if (price == 0.0) return;  // no price -> drop the order (mirrors Python guard)

        const double fill_price = (o.direction == Direction::Buy)
                                      ? price * (1.0 + slippage_pct_)
                                      : price * (1.0 - slippage_pct_);

        FillEvent f;
        f.timeindex  = data_.timeindex();
        f.ticker     = o.ticker;
        f.quantity   = o.quantity;
        f.direction  = o.direction;
        f.fill_cost  = fill_price;
        f.commission = std::max(0.005 * o.quantity, 1.00);
        events.push(f);
    }

private:
    const DataHandler& data_;
    double             slippage_pct_;
};
