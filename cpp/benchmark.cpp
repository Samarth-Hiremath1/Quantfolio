#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sys/resource.h>
#include "engine.hpp"

// Honest benchmark, matched to the Python harness workload:
//  * synthetic geometric-Brownian prices (mean 0.0004, vol 0.012 per bar),
//    same statistical process the Python benchmark used
//  * scaling sweep 7 -> 200 securities at T = 252 bars
//  * bars/sec AND events/sec reported separately
//  * 5 repetitions per config; median plus min/max
//  * peak process RSS via getrusage (REAL RSS, not an allocator counter)

using Clock = std::chrono::steady_clock;

// Generate [n_tickers][n_bars] adj_close, seeded for reproducibility.
static std::vector<std::vector<double>>
make_prices(int n_tickers, int n_bars, uint64_t seed) {
    std::vector<std::vector<double>> px(n_tickers, std::vector<double>(n_bars));
    for (int t = 0; t < n_tickers; ++t) {
        std::mt19937_64 rng(seed + static_cast<uint64_t>(t));
        std::normal_distribution<double> nd(0.0004, 0.012);
        double logp = std::log(100.0);
        for (int b = 0; b < n_bars; ++b) {
            logp += nd(rng);
            px[t][b] = std::exp(logp);
        }
    }
    return px;
}

// Peak resident set size of this process, in megabytes.
// ru_maxrss is a high-water mark: bytes on macOS, kilobytes on Linux.
static double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

struct Stat { double median, lo, hi; };
static Stat summarize(std::vector<double> xs) {
    std::sort(xs.begin(), xs.end());
    return { xs[xs.size() / 2], xs.front(), xs.back() };
}

int main() {
    const int    n_bars = 252;
    const int    reps   = 5;
    const std::vector<int> universe = {7, 25, 50, 100, 200};

    std::printf("QuantFolio C++ engine benchmark\n");
    std::printf("T=%d bars, reps=%d, seed=42, -O3, single process\n\n", n_bars, reps);
    std::printf("%-6s %12s %12s %14s %14s %10s %8s\n",
                "secs", "bars/s(med)", "[min..max]",
                "events/s(med)", "ev/bar", "wall_ms", "trades");

    double peak_overall = 0.0;

    for (int n : universe) {
        auto prices = make_prices(n, n_bars, /*seed=*/42);

        std::vector<double> bars_per_s, events_per_s, wall_ms;
        long total_events = 0, trades = 0;

        for (int rep = 0; rep < reps; ++rep) {
            // fresh copy per rep so each run does identical work
            auto px = prices;
            BacktestingEngine engine(DataHandler(std::move(px)));

            auto t0 = Clock::now();
            auto r  = engine.run();
            auto t1 = Clock::now();

            double sec = std::chrono::duration<double>(t1 - t0).count();
            long ev = r.bars + r.signals + r.orders + r.fills;
            total_events = ev;
            trades = r.fills;

            bars_per_s.push_back(r.bars / sec);
            events_per_s.push_back(ev / sec);
            wall_ms.push_back(sec * 1000.0);
        }

        Stat b = summarize(bars_per_s);
        Stat e = summarize(events_per_s);
        Stat w = summarize(wall_ms);
        peak_overall = std::max(peak_overall, peak_rss_mb());

        std::printf("%-6d %12.0f %6.0f..%-6.0f %14.0f %14.2f %10.3f %8ld\n",
                    n, b.median, b.lo, b.hi, e.median,
                    static_cast<double>(total_events) / n_bars, w.median, trades);
    }

    std::printf("\nPeak process RSS (getrusage high-water): %.2f MB\n", peak_overall);

    // Robust headline config: 7 securities x 2520 bars, 11 reps. Longer wall
    // time (~100us+) so the number is not timer-resolution-limited like the
    // 7x252 row above. Matched to the Python horizon point of ~5,093 bars/sec
    // at 7 assets x 2520 bars (untraced).
    {
        const int T = 2520, R = 11;
        auto prices = make_prices(7, T, 42);
        std::vector<double> bps;
        for (int rep = 0; rep < R; ++rep) {
            auto px = prices;
            BacktestingEngine engine(DataHandler(std::move(px)));
            auto t0 = Clock::now();
            auto r  = engine.run();
            auto t1 = Clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            bps.push_back(r.bars / sec);
        }
        Stat s = summarize(bps);
        std::printf("\nHeadline (7 secs x %d bars, %d reps): median %.0f bars/s "
                    "[%.0f..%.0f]\n", T, R, s.median, s.lo, s.hi);
        std::printf("Python reference (7 secs x %d bars, untraced): ~5,093 bars/s\n", T);
        std::printf("Speedup at this matched config: ~%.0fx\n", s.median / 5093.0);
    }
    return 0;
}
