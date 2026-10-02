# C++ engine

A standalone C++17 port of the backtesting core in `backtesting/`. It has no Python dependency. The file layout matches the Python modules one to one, so the two can be read side by side.

| File | Python counterpart |
|---|---|
| `event.hpp` | `backtesting/events.py` |
| `data_handler.hpp` | `backtesting/data_handler.py` |
| `strategy.hpp` | `backtesting/strategy.py` |
| `portfolio.hpp` | `backtesting/portfolio.py` |
| `execution.hpp` | `backtesting/execution.py` |
| `engine.hpp` | `backtesting/engine.py` |

## Build

```bash
cd cpp
cmake -B build && cmake --build build
./build/backtest ../tests/fixtures/raw_ohlcv_sample.csv
./build/benchmark
```

Or without CMake:

```bash
clang++ -std=c++17 -O3 -DNDEBUG main.cpp -o backtest
clang++ -std=c++17 -O3 -DNDEBUG benchmark.cpp -o benchmark
```

## The loop

The outer loop pushes one `MarketEvent` per bar. The inner loop then drains the queue until it is empty, and only after that does time advance. A market event can cascade into a signal, an order and a fill inside the same timestamp, and all of them resolve against the current bar's price. A fill can never see the next bar. That ordering is the whole point of the event-driven design, and it is the same in both languages.

## Design choices

**Events are a `std::variant` of plain structs.** The alternative was a base class with virtual methods held in a `std::queue<std::unique_ptr<Event>>`, which is the closest translation of the Python class hierarchy. I chose the variant because events then live by value inside the queue's buffer. Pushing one copies about 40 bytes. There is no heap allocation per event and no vtable lookup. The cost is that every event takes the size of the largest member (`FillEvent`), and adding a fifth event type means touching every `std::visit`. With four fixed types that is a fair trade.

**Tickers and directions are integers and enums, not strings.** The Python events carry `'AAPL'`, `'BUY'`, `'SMART'`. A `std::string` field would allocate on every event, so the C++ structs use an index into the universe and small enums instead.

**No owning pointers in the run loop.** `DataHandler` owns the price matrix. The strategy, portfolio and execution handler hold a `const DataHandler&` and never copy it.

**`std::queue` over `std::deque`.** The loop only needs FIFO push and pop. The queue is a member of the engine, so its buffer is reused across bars instead of being rebuilt.

**Rolling sums for the moving averages.** Each ticker keeps a running fast sum and slow sum. A new bar adds the new price and subtracts the one leaving the window, which is read straight from the price matrix. That makes the SMA update O(1) per ticker per bar and avoids a second copy of the history.

**`-O3`.** The loop is compute-bound and there is no code-size constraint. I expect the gain over `-O2` to be small here, since queue and dispatch overhead dominate over arithmetic the vectorizer could widen. I have not measured the two against each other.

## Results

Apple M1 Pro, clang 15, `-O3`, single thread. Synthetic geometric-Brownian prices, seed 42, 252 bars, five runs per configuration.

| Securities | Bars/sec (median) | Min – max | Events/sec (median) | Events per bar |
|---:|---:|---:|---:|---:|
| 7 | 26.1M | 22.9M – 27.1M | 39.4M | 1.5 |
| 25 | 9.0M | 8.2M – 9.3M | 26.5M | 2.9 |
| 50 | 4.6M | 4.4M – 4.6M | 22.4M | 4.9 |
| 100 | 1.9M | 1.86M – 1.92M | 16.6M | 8.8 |
| 200 | 0.92M | 0.78M – 0.93M | 15.0M | 16.2 |

Throughput is not flat. Bars per second fall about 28x between 7 and 200 securities, because the strategy and the mark-to-market both walk every ticker on every bar. Events per second fall more gently, about 2.6x, since a larger universe also produces more crossovers per bar.

The 7- and 25-security rows finish in 10 to 30 microseconds, which is close to timer resolution. Treat them as rough. A longer run is steadier but still moves between invocations: 7 securities over 2,520 bars, eleven runs each, gave medians of 14.0M, 15.5M and 17.6M bars/sec on three separate invocations.

Peak resident set size, read from `getrusage`, was 1.8 MB for the sweep and 2.8 MB with the long run included. This is whole-process RSS.

### Against the Python engine

The Python engine manages about 5,100 bars/sec on the same 7-security, 2,520-bar workload. The C++ port is between 2,700x and 3,500x faster, depending on the run. That figure needs context. The Python `DataHandler` iterates with `DataFrame.iterrows()` and builds a dict per bar, which is slow even by Python standards. So the gap measures compiled code, value-typed events and flat arrays against an interpreter plus pandas row construction. A NumPy-backed Python loop would close a good part of it.

### On the sample CSV

`./backtest` on the bundled fixture reports zero trades. That is correct. The file holds 43 bars and the slow average needs 50, so no crossover can form. The Python engine behaves the same way on this file.

## Limitations

- `std::deque` still allocates in chunks. A fixed-capacity ring buffer would remove the last allocations from the loop.
- Prices are stored as a vector of vectors. One flat `N x T` array would be kinder to the cache.
- Money is `double`. A production system would use fixed-point.
- Fills are instant and complete, at the bar's close plus 0.05% slippage. There are no partial fills and no latency model.
- Position sizing is a fixed 100 shares.
- The benchmark does not pin cores or discard warm-up runs.
