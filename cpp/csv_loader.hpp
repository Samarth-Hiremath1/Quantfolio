#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

// Loads the same long-format CSV the Python side uses:
//   trade_date,open,high,low,close,adj_close,volume,ticker
// and returns a dense [ticker][bar] adj_close matrix aligned on the sorted set
// of dates. Missing (date,ticker) cells are forward-filled, then back-filled,
// so every ticker has a value on every date (the engine assumes a rectangular
// grid). Also returns the ticker symbols in column order.
struct LoadedData {
    std::vector<std::vector<double>> prices;  // [ticker][bar]
    std::vector<std::string>         tickers;
    int n_bars = 0;
};

inline LoadedData load_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string line;
    std::getline(in, line);  // header

    // (ticker -> (date -> adj_close))
    std::map<std::string, std::map<std::string, double>> byTicker;
    std::set<std::string> allDates;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string date, o, h, l, c, adj, vol, tk;
        std::getline(ss, date, ',');
        std::getline(ss, o, ',');
        std::getline(ss, h, ',');
        std::getline(ss, l, ',');
        std::getline(ss, c, ',');
        std::getline(ss, adj, ',');
        std::getline(ss, vol, ',');
        std::getline(ss, tk, ',');
        if (date.empty() || tk.empty() || adj.empty()) continue;
        byTicker[tk][date] = std::stod(adj);
        allDates.insert(date);
    }

    LoadedData out;
    std::vector<std::string> dates(allDates.begin(), allDates.end());  // sorted
    out.n_bars = static_cast<int>(dates.size());

    for (auto& [tk, series] : byTicker) {
        out.tickers.push_back(tk);
        std::vector<double> col(dates.size(), 0.0);
        double last = 0.0;
        for (size_t i = 0; i < dates.size(); ++i) {
            auto it = series.find(dates[i]);
            if (it != series.end()) last = it->second;
            col[i] = last;  // forward fill
        }
        // back-fill any leading zeros with the first real value
        double firstReal = 0.0;
        for (double v : col) if (v != 0.0) { firstReal = v; break; }
        for (double& v : col) if (v == 0.0) v = firstReal;
        out.prices.push_back(std::move(col));
    }
    return out;
}
