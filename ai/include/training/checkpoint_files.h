#pragma once
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// File names in a goes_train checkpoint directory:
//   <arch>_config.json        game config + model config (what goes_server matches /move requests on)
//   <arch>_<iter>.pt          the model exported at the end of that iteration, kept every
//                             --saves-every iterations and at iteration --iterations (what goes_server
//                             uses)
//   <arch>_<iter>_traj.json   that iteration's self-play games (SelfPlayRunner's trajectory format)
//   trainer/                  what --resume needs: model.pt, weight_average.pt, optimizer.pt, state.json
// <iter> is zero-padded to 6 digits.

inline constexpr std::string_view kConfigSuffix = "_config.json";
inline constexpr std::string_view kModelSuffix = ".pt";
inline constexpr std::string_view kTrajectorySuffix = "_traj.json";

inline std::string config_file_name(std::string_view arch) { return std::string(arch) + std::string(kConfigSuffix); }

// <arch>_<iter><suffix>.
inline std::string iter_file_name(std::string_view arch, int iter, std::string_view suffix) {
    std::ostringstream oss;
    oss << arch << "_" << std::setfill('0') << std::setw(6) << iter << suffix;
    return oss.str();
}

// The iter of a file named exactly iter_file_name(arch, iter, suffix), else nullopt.
inline std::optional<int> parse_iter_file_name(std::string_view name, std::string_view arch, std::string_view suffix) {
    if (name.size() <= arch.size() + 1 + suffix.size() || !name.starts_with(arch) || !name.ends_with(suffix))
        return std::nullopt;
    std::string_view digits = name.substr(arch.size() + 1, name.size() - arch.size() - 1 - suffix.size());
    int iter = 0;
    auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), iter);
    if (ec != std::errc() || end != digits.data() + digits.size() || iter_file_name(arch, iter, suffix) != name)
        return std::nullopt;
    return iter;
}

// The iterations with an iter_file_name(arch, iter, suffix) file directly in `dir`, ascending.
inline std::vector<int> iter_files(const std::filesystem::path& dir, std::string_view arch, std::string_view suffix) {
    std::vector<int> iters;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.is_regular_file())
            if (auto iter = parse_iter_file_name(e.path().filename().string(), arch, suffix)) iters.push_back(*iter);
    std::sort(iters.begin(), iters.end());
    return iters;
}

// The <arch>_config.json in `dir` (the first entry whose name ends in "_config.json"), or nullopt if
// `dir` doesn't exist or has none.
inline std::optional<std::filesystem::path> find_config_json(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) return std::nullopt;
    for (auto& e : std::filesystem::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (name.size() > kConfigSuffix.size() && name.ends_with(kConfigSuffix)) return e.path();
    }
    return std::nullopt;
}
