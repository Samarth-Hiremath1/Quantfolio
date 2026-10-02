#include <cstdio>
#include <string>
#include "csv_loader.hpp"
#include "engine.hpp"

// Runs the C++ engine on the same real CSV the Python version uses and prints
// final value, total return, and trade (fill) count.
//
// Usage: ./backtest [path-to-raw_ohlcv.csv]
int main(int argc, char** argv) {
    const std::string path =
        (argc > 1) ? argv[1] : "../tests/fixtures/raw_ohlcv_sample.csv";

    LoadedData data;
    try {
        data = load_csv(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    std::printf("Loaded %zu tickers x %d bars from %s\n",
                data.tickers.size(), data.n_bars, path.c_str());
    std::printf("Tickers:");
    for (auto& t : data.tickers) std::printf(" %s", t.c_str());
    std::printf("\n");

    const double capital = 100000.0;
    BacktestingEngine engine(DataHandler(std::move(data.prices)), capital);
    auto r = engine.run();

    std::printf("\n=== Backtest complete ===\n");
    std::printf("Initial capital : $%.2f\n", capital);
    std::printf("Final value     : $%.2f\n", r.final_value);
    std::printf("Total return    : %.4f%%\n", r.total_return_pct);
    std::printf("Bars processed  : %ld\n", r.bars);
    std::printf("Signals/Orders/Fills : %ld / %ld / %ld\n",
                r.signals, r.orders, r.fills);
    std::printf("Trade count     : %ld\n", r.fills);
    return 0;
}
