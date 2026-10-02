#pragma once
#include <variant>
#include <cstdint>

// ---------------------------------------------------------------------------
// Event types. Mirror of backtesting/events.py.
//
// DESIGN CHOICE: std::variant of value types, NOT a base class + virtual
// dispatch. See cpp/README.md. In one line: events live by value in
// the queue (no heap allocation per event, no vtable indirection), and the
// tag is the variant's discriminant instead of a `self.type` string.
//
// The Python events carry string fields (strategy_id, exchange, ticker symbol).
// Here those become integers / enums so an event is a small trivially-copyable
// POD -- pushing one into the queue copies a few machine words and never
// touches the heap. That is deliberate and is the main per-event win.
// ---------------------------------------------------------------------------

enum class EventType : uint8_t { Market, Signal, Order, Fill };
enum class Direction : uint8_t { Buy, Sell };
enum class SignalType : uint8_t { Long, Short };

// Empty like Python's MarketEvent() -- it only signals "a new bar exists".
struct MarketEvent {
    static constexpr EventType type = EventType::Market;
};

struct SignalEvent {
    static constexpr EventType type = EventType::Signal;
    int        ticker;        // index into the universe (Python used the symbol string)
    long       timeindex;     // bar index (Python used the datetime)
    SignalType signal_type;
    double     strength;      // typically 1.0, mirrors Python
};

struct OrderEvent {
    static constexpr EventType type = EventType::Order;
    int       ticker;
    int       quantity;       // fixed 100-share lots
    Direction direction;
    // order_type is always 'MKT' in the Python version, so it is implicit here.
};

struct FillEvent {
    static constexpr EventType type = EventType::Fill;
    long      timeindex;
    int       ticker;
    int       quantity;
    Direction direction;
    double    fill_cost;      // per-share price actually paid, after slippage
    double    commission;
    // exchange is always 'SMART' in the Python version, so it is implicit here.
};

using Event = std::variant<MarketEvent, SignalEvent, OrderEvent, FillEvent>;

// Helper for std::visit dispatch: overloaded{lambdas...}.
template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
