// Part 1: brute-force magic square solver (C++17 + OpenMP).
//
// Goal per size n: find ONE magic square by enumerating permutations of 1..n^2
// in lexicographic order, writing each into an n x n square and checking rows,
// columns and both main diagonals. No pruning on partial sums.
//
// Parallelisation: the lexicographic permutation space is cut into tasks by
// fixing the first D cells (task t = t-th ordered D-prefix, obtained by
// unranking). Each task enumerates the remaining (N-D)! orderings with
// std::next_permutation. OpenMP hands out tasks in increasing order
// (schedule(dynamic,1)); in "first" mode threads skip/abort every task whose
// index is larger than the best task that already contains a solution, so the
// result is deterministic: the lexicographically smallest magic square.
//
// Modes
//   first : stop at the first (lexicographically smallest) solution.   [default]
//   full  : enumerate everything (or the first --max-tasks tasks); counts solutions.
//   timed : enumerate for --seconds seconds (throughput measurement).
// --timeout S (first mode) gives up after S seconds (status=timeout).
//
// One CSV row is appended per run (--csv).

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

typedef std::uint64_t u64;
typedef std::int64_t i64;

static inline bool is_magic(const int* a, int n, int target) {
    for (int r = 0; r < n; ++r) {            // rows (row 0 fails for almost all permutations)
        int s = 0; const int* row = a + r * n;
        for (int c = 0; c < n; ++c) s += row[c];
        if (s != target) return false;
    }
    for (int c = 0; c < n; ++c) {            // columns
        int s = 0;
        for (int r = 0; r < n; ++r) s += a[r * n + c];
        if (s != target) return false;
    }
    int d1 = 0, d2 = 0;                      // diagonals
    for (int i = 0; i < n; ++i) { d1 += a[i * n + i]; d2 += a[i * n + (n - 1 - i)]; }
    return d1 == target && d2 == target;
}

// t-th (0-based) ordered D-prefix of 1..N in lexicographic order.
static void unrank_prefix(int N, int D, u64 t, int* out) {
    std::vector<int> avail(N);
    std::iota(avail.begin(), avail.end(), 1);
    for (int k = 0; k < D; ++k) {
        u64 block = 1;
        for (int j = k + 1; j < D; ++j) block *= (u64)(N - j);
        u64 idx = t / block; t %= block;
        out[k] = avail[idx];
        avail.erase(avail.begin() + idx);
    }
}

int main(int argc, char** argv) {
    int n = 3, threads = omp_get_max_threads(), depth = -1, repeat = 1;
    std::string mode = "first", csv = "../../plots/plot-part1.csv";
    double seconds = 10.0, timeout = 0.0;
    u64 max_tasks = 0;
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() { if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); } return argv[++i]; };
        if (a == "--n") n = std::atoi(need());
        else if (a == "--mode") mode = need();
        else if (a == "--seconds") seconds = std::atof(need());
        else if (a == "--timeout") timeout = std::atof(need());
        else if (a == "--threads") threads = std::atoi(need());
        else if (a == "--prefix-depth") depth = std::atoi(need());
        else if (a == "--max-tasks") max_tasks = std::strtoull(need(), nullptr, 10);
        else if (a == "--csv") csv = need();
        else if (a == "--repeat") repeat = std::atoi(need());
        else if (a == "--quiet") quiet = true;
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (n < 1 || n > 5) { std::cerr << "n must be in 1..5\n"; return 2; }
    if (mode != "first" && mode != "full" && mode != "timed") { std::cerr << "mode must be first|full|timed\n"; return 2; }
    const int N = n * n, target = n * (N + 1) / 2;
    if (depth < 0) depth = (mode == "first") ? std::min(N, 5) : std::min(N, 2);
    depth = std::max(1, std::min(depth, N));
    omp_set_num_threads(threads);

    // number of tasks = N!/(N-D)!  (clamped), optionally capped by --max-tasks
    long double tl = 1; for (int j = 0; j < depth; ++j) tl *= (N - j);
    u64 ntasks_u = tl > 4e18L ? (u64)4e18 : (u64)tl;
    if (max_tasks && max_tasks < ntasks_u) ntasks_u = max_tasks;
    const i64 ntasks = (i64)ntasks_u;
    const bool whole_space_tasks = (max_tasks == 0) || (long double)ntasks >= tl;
    u64 perms_per_task = 1; for (int k = 2; k <= N - depth && k <= 20; ++k) perms_per_task *= k;
    const double log10_total = std::lgamma(N + 1.0) / std::log(10.0);
    const double limit = (mode == "timed") ? seconds : (mode == "first" ? timeout : 0.0);

    for (int rep = 0; rep < repeat; ++rep) {
        std::atomic<bool> stop(false);
        std::atomic<i64> best_task(INT64_MAX);
        u64 checked = 0, solutions = 0; i64 tasks_done = 0;
        i64 sol_task = INT64_MAX; u64 sol_idx = 0; std::vector<int> sol;
        double t0 = omp_get_wtime();

        #pragma omp parallel
        {
            std::vector<int> arr(N), pre(depth);
            std::vector<char> used(N + 1);
            u64 my_checked = 0, my_sol = 0; i64 my_tasks = 0;
            i64 my_best = INT64_MAX; u64 my_idx = 0; std::vector<int> my_sq;

            #pragma omp for schedule(dynamic, 1)
            for (i64 t = 0; t < ntasks; ++t) {
                if (stop.load(std::memory_order_relaxed)) continue;
                if (mode == "first" && t > best_task.load(std::memory_order_relaxed)) continue;
                unrank_prefix(N, depth, (u64)t, pre.data());
                std::fill(used.begin(), used.end(), 0);
                for (int d = 0; d < depth; ++d) { arr[d] = pre[d]; used[pre[d]] = 1; }
                int pos = depth;
                for (int v = 1; v <= N; ++v) if (!used[v]) arr[pos++] = v;   // sorted remainder
                bool finished = true; u64 idx = 0;
                do {
                    ++my_checked;
                    if (is_magic(arr.data(), n, target)) {
                        ++my_sol;
                        if (t < my_best) { my_best = t; my_idx = idx; my_sq.assign(arr.begin(), arr.end()); }
                        if (mode == "first") {
                            i64 cur = best_task.load();
                            while (t < cur && !best_task.compare_exchange_weak(cur, t)) {}
                            finished = true; break;
                        }
                    }
                    ++idx;
                    if ((idx & 0xFFFFF) == 0) {
                        if (stop.load(std::memory_order_relaxed)) { finished = false; break; }
                        if (mode == "first" && best_task.load(std::memory_order_relaxed) < t) { finished = false; break; }
                        if (limit > 0 && omp_get_wtime() - t0 >= limit) { stop.store(true); finished = false; break; }
                    }
                } while (std::next_permutation(arr.begin() + depth, arr.end()));
                if (finished) ++my_tasks;
            }
            #pragma omp critical
            {
                checked += my_checked; solutions += my_sol; tasks_done += my_tasks;
                if (my_best < sol_task) { sol_task = my_best; sol_idx = my_idx; sol = my_sq; }
            }
        }
        double elapsed = omp_get_wtime() - t0;

        const bool found = !sol.empty();
        std::string status;
        if (mode == "first") status = found ? "found" : (stop.load() ? "timeout" : "exhausted");
        else if (mode == "timed") status = (tasks_done == ntasks && whole_space_tasks) ? "exhausted" : "partial";
        else status = whole_space_tasks ? "exhausted" : "partial";
        // lexicographic rank (0-based) of the first solution: t*(N-D)! + idx
        long double rank = found ? (long double)sol_task * (long double)perms_per_task + (long double)sol_idx : -1;
        double throughput = elapsed > 0 ? checked / elapsed : 0.0;

        bool need_header;
        { std::ifstream f(csv); need_header = !f.good() || f.peek() == std::ifstream::traits_type::eof(); }
        std::ofstream out(csv, std::ios::app);
        if (need_header)
            out << "n,mode,threads,prefix_depth,tasks_total,tasks_done,permutations_checked,log10_total_permutations,"
                   "elapsed_seconds,perms_per_second,solutions_found,first_solution_rank,status\n";
        out.setf(std::ios::fixed); out.precision(9);
        out << n << ',' << mode << ',' << threads << ',' << depth << ',' << ntasks << ',' << tasks_done << ','
            << checked << ',' << log10_total << ',' << elapsed << ',' << throughput << ',' << solutions << ','
            << (found ? (u64)rank : (u64)0) << ',' << status << '\n';

        if (!quiet) {
            std::printf("n=%d mode=%s threads=%d depth=%d checked=%llu time=%.6fs throughput=%.3e/s status=%s",
                        n, mode.c_str(), threads, depth, (unsigned long long)checked, elapsed, throughput, status.c_str());
            if (found) std::printf(" rank=%llu", (unsigned long long)rank);
            std::printf("\n");
            if (found && rep == 0) {
                for (int r = 0; r < n; ++r) { for (int c = 0; c < n; ++c) std::printf("%3d ", sol[r * n + c]); std::printf("\n"); }
            }
        }
    }
    return 0;
}
