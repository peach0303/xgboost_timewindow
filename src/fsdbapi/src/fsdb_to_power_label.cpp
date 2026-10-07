#ifdef NOVAS_FSDB
#undef NOVAS_FSDB
#endif

#include "ffrAPI.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

namespace {

struct Options {
    std::string fsdb_path;
    std::string output_tsv;
    std::string output_meta;
    std::string power_signal;
    std::string list_signals;
    double start = 0.0;
    double end = 0.0;
    double step = 0.0;
    bool has_start = false;
    bool has_end = false;
    bool end_inclusive = false;
    bool verbose = false;
    fsdbXTagType xtag_type = FSDB_XTAG_TYPE_HL;
};

struct Signal {
    fsdbVarIdcode idcode = 0;
    std::string full_name;
    byte_T vc_dt = 0;
    byte_T bytes_per_bit = 0;
    byte_T type = 0;
    ushort_T dtidcode = 0;
};

struct TreeContext {
    std::vector<std::string> scope_stack;
    std::vector<Signal> signals;
    std::string sep = "/";
    bool list_all_signals = false;  // when true, collect all signals regardless of type
};

struct Event {
    double time = 0.0;
    double value = 0.0;
};

static void die(const std::string &msg) {
    throw std::runtime_error(msg);
}

static std::string json_escape(const std::string &s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                os << buf;
            } else {
                os << c;
            }
        }
    }
    return os.str();
}

static std::string dirname_of(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static void mkdir_p(const std::string &path) {
    if (path.empty() || path == "." || path == "/") return;
    std::string cur;
    size_t i = 0;
    if (path[0] == '/') { cur = "/"; i = 1; }
    while (i <= path.size()) {
        size_t slash = path.find('/', i);
        std::string part = path.substr(i, slash == std::string::npos ? std::string::npos : slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur += "/";
            cur += part;
            if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
                die("cannot mkdir " + cur + ": " + std::strerror(errno));
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
}

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static uint64_t tag64_to_u64(const fsdbTag64 &t) {
    return (static_cast<uint64_t>(t.H) << 32) | static_cast<uint64_t>(t.L);
}

static fsdbTag64 u64_to_tag64(uint64_t v) {
    fsdbTag64 t;
    std::memset(&t, 0, sizeof(t));
    t.H = static_cast<uint_T>(v >> 32);
    t.L = static_cast<uint_T>(v & 0xffffffffu);
    return t;
}

static bool is_numeric_var(const fsdbTreeCBDataVar *var) {
    if (!var || var->is_dummy_var || var->is_void) return false;
    switch (var->vc_dt) {
    case FSDB_VC_DT_BYTE:       // 1
    case FSDB_VC_DT_SHORT:      // 2
    case FSDB_VC_DT_INT:        // 3
    case FSDB_VC_DT_LONG:       // 4
    case FSDB_VC_DT_FLOAT:      // 5
    case FSDB_VC_DT_DOUBLE:     // 6
    case FSDB_VC_DT_HL_INT:     // 7 - PrimeTime PX may use this
    case FSDB_VC_DT_PHYSICAL:   // 8 - PrimeTime PX power physical quantity
        return true;
    default:
        break;
    }
    // Fallback: vc_dt may be UNKNOWN(0) but dtidcode reveals numeric type
    return var->dtidcode == FSDB_DT_VERILOG_REAL
        || var->dtidcode == FSDB_DT_FLOAT
        || var->dtidcode == FSDB_DT_DOUBLE
        || var->dtidcode == FSDB_DT_HL_INT;
}

static std::string current_scope(const TreeContext &ctx) {
    std::ostringstream os;
    for (size_t i = 0; i < ctx.scope_stack.size(); ++i) {
        if (i) os << ctx.sep;
        os << ctx.scope_stack[i];
    }
    return os.str();
}

static bool_T tree_cb(fsdbTreeCBType cb_type, void *client_data, void *tree_cb_data) {
    TreeContext *ctx = static_cast<TreeContext *>(client_data);
    switch (cb_type) {
    case FSDB_TREE_CBT_SCOPE: {
        auto *scope = static_cast<fsdbTreeCBDataScope *>(tree_cb_data);
        if (scope && scope->name) ctx->scope_stack.emplace_back(scope->name);
        break;
    }
    case FSDB_TREE_CBT_UPSCOPE:
        if (!ctx->scope_stack.empty()) ctx->scope_stack.pop_back();
        break;
    case FSDB_TREE_CBT_VAR: {
        auto *var = static_cast<fsdbTreeCBDataVar *>(tree_cb_data);
        // Skip dummy/void vars, but accept ALL real signals.
        // is_numeric_var is only a hint — PrimeTime PX may use types
        // not in the whitelist. decode_numeric has a double fallback.
        if (!var || var->is_dummy_var || var->is_void) break;
        std::string scope = current_scope(*ctx);
        std::string local = var->name ? var->name : "";
        Signal sig;
        sig.idcode = var->u.idcode;
        sig.full_name = scope.empty() ? local : scope + ctx->sep + local;
        sig.vc_dt = var->vc_dt;
        sig.bytes_per_bit = var->bytes_per_bit;
        sig.type = var->type;
        sig.dtidcode = var->dtidcode;
        ctx->signals.push_back(sig);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static double parse_double(const std::string &s, const std::string &name) {
    char *end = nullptr;
    errno = 0;
    double v = std::strtod(s.c_str(), &end);
    if (errno || !end || *end) die("bad number for " + name + ": " + s);
    return v;
}

static double decode_numeric(byte_T vc_dt, const byte_T *vc, ushort_T dtidcode) {
    if (!vc) return std::numeric_limits<double>::quiet_NaN();

    // When vc_dt is UNKNOWN(0) but dtidcode reveals the true format,
    // decode based on dtidcode.
    if (vc_dt == FSDB_VC_DT_UNKNOWN) {
        switch (dtidcode) {
        case 9: { // FSDB_DT_EPIC_ANALOG — PrimeTime PX power as double
            double x = 0.0;
            std::memcpy(&x, vc, sizeof(x));
            if (std::isfinite(x)) return x;
            break;
        }
        case 8: { // FSDB_DT_EPIC_DIGITAL — try int32
            int32_t x = 0;
            std::memcpy(&x, vc, sizeof(x));
            return static_cast<double>(x);
        }
        default:
            break;
        }
        // Generic fallback for UNKNOWN: try float, then double, then int64
        {
            float xf = 0.0f;
            std::memcpy(&xf, vc, sizeof(xf));
            if (std::isfinite(xf) && std::abs(xf) > 1e-45f && std::abs(xf) < 1e+38f)
                return static_cast<double>(xf);
        }
        {
            double xd = 0.0;
            std::memcpy(&xd, vc, sizeof(xd));
            if (std::isfinite(xd)) return xd;
        }
        {
            int64_t xi = 0;
            std::memcpy(&xi, vc, sizeof(xi));
            return static_cast<double>(xi);
        }
    }

    switch (vc_dt) {
    case FSDB_VC_DT_BYTE: {
        int8_t x = 0; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_SHORT: {
        int16_t x = 0; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_INT: {
        int32_t x = 0; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_LONG: {
        int64_t x = 0; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_HL_INT: {
        int64_t x = 0; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_FLOAT: {
        float x = 0.0f; std::memcpy(&x, vc, sizeof(x)); return static_cast<double>(x);
    }
    case FSDB_VC_DT_DOUBLE: {
        double x = 0.0; std::memcpy(&x, vc, sizeof(x)); return x;
    }
    case FSDB_VC_DT_PHYSICAL: {
        int64_t xi = 0;
        std::memcpy(&xi, vc, sizeof(xi));
        double xd = 0.0;
        std::memcpy(&xd, vc, sizeof(xd));
        if (std::abs(xi) >= 0 && std::abs(xi) <= 1000000000000000LL
            && (!std::isfinite(xd) || std::abs(xd) < 1e-300 || std::abs(xd) > 1e+300)) {
            return static_cast<double>(xi);
        }
        if (std::isfinite(xd)) return xd;
        return static_cast<double>(xi);
    }
    default:
        // Generic fallback: try float, double, int64
        {
            float xf = 0.0f;
            std::memcpy(&xf, vc, sizeof(xf));
            if (std::isfinite(xf) && std::abs(xf) > 1e-45f && std::abs(xf) < 1e+38f)
                return static_cast<double>(xf);
        }
        {
            double xd = 0.0;
            std::memcpy(&xd, vc, sizeof(xd));
            if (std::isfinite(xd)) return xd;
        }
        {
            int64_t xi = 0;
            std::memcpy(&xi, vc, sizeof(xi));
            return static_cast<double>(xi);
        }
    }
}

static void apply_auto_time_range(ffrObject *fsdb_obj, Options &opt) {
    opt.xtag_type = fsdb_obj->ffrGetXTagType();
    if (opt.verbose) {
        std::cerr << "[time] XTag type: ";
        switch (opt.xtag_type) {
        case FSDB_XTAG_TYPE_L:  std::cerr << "L (32-bit int)\n"; break;
        case FSDB_XTAG_TYPE_HL: std::cerr << "HL (64-bit int)\n"; break;
        case FSDB_XTAG_TYPE_FLOAT: std::cerr << "FLOAT\n"; break;
        case FSDB_XTAG_TYPE_DOUBLE: std::cerr << "DOUBLE\n"; break;
        default: std::cerr << static_cast<int>(opt.xtag_type) << "\n"; break;
        }
    }
    if (!opt.has_start) {
        switch (opt.xtag_type) {
        case FSDB_XTAG_TYPE_FLOAT: {
            fsdbTagFloat tf = 0.0f;
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMinFsdbTagFloat(&tf))
                die("cannot auto-detect FSDB min time tag; pass --start explicitly");
            opt.start = static_cast<double>(tf);
            break;
        }
        case FSDB_XTAG_TYPE_DOUBLE: {
            fsdbTagDouble td = 0.0;
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMinFsdbTagDouble(&td))
                die("cannot auto-detect FSDB min time tag; pass --start explicitly");
            opt.start = td;
            break;
        }
        default: {
            fsdbTag64 t64;
            std::memset(&t64, 0, sizeof(t64));
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMinFsdbTag64(&t64))
                die("cannot auto-detect FSDB min time tag; pass --start explicitly");
            opt.start = static_cast<double>(tag64_to_u64(t64));
            break;
        }
        }
    }
    if (!opt.has_end) {
        switch (opt.xtag_type) {
        case FSDB_XTAG_TYPE_FLOAT: {
            fsdbTagFloat tf = 0.0f;
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMaxFsdbTagFloat(&tf))
                die("cannot auto-detect FSDB max time tag; pass --end explicitly");
            opt.end = static_cast<double>(tf);
            break;
        }
        case FSDB_XTAG_TYPE_DOUBLE: {
            fsdbTagDouble td = 0.0;
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMaxFsdbTagDouble(&td))
                die("cannot auto-detect FSDB max time tag; pass --end explicitly");
            opt.end = td;
            break;
        }
        default: {
            fsdbTag64 t64;
            std::memset(&t64, 0, sizeof(t64));
            if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMaxFsdbTag64(&t64))
                die("cannot auto-detect FSDB max time tag; pass --end explicitly");
            opt.end = static_cast<double>(tag64_to_u64(t64));
            break;
        }
        }
    }
    if (opt.end <= opt.start) die("--end must be larger than --start");
    if (opt.verbose) std::cerr << "[time] range: " << opt.start << " .. " << opt.end << " step=" << opt.step << "\n";
}

static std::vector<double> build_sample_times(const Options &opt) {
    std::vector<double> times;
    double t = opt.start;
    const double eps = 1e-9;
    while (t + eps < opt.end || (opt.end_inclusive && std::abs(t - opt.end) < eps)) {
        times.push_back(t);
        t += opt.step;
        if (times.size() > 100'000'000) die("too many sample times (>100M)");
    }
    return times;
}

static const Signal *select_signal(const std::vector<Signal> &signals, const std::string &query) {
    std::vector<const Signal *> matches;
    for (const auto &s : signals) {
        if (s.full_name == query || ends_with(s.full_name, "/" + query) || ends_with(s.full_name, query)) {
            matches.push_back(&s);
        }
    }
    if (matches.empty()) return nullptr;
    if (matches.size() > 1) {
        std::cerr << "ambiguous --power-signal, matches:\n";
        for (const auto *s : matches) std::cerr << "  " << s->full_name << " idcode=" << s->idcode << "\n";
        die("ambiguous --power-signal: " + query);
    }
    return matches[0];
}

static double xtag_to_double(const fsdbXTag &xt, fsdbXTagType xtype) {
    switch (xtype) {
    case FSDB_XTAG_TYPE_FLOAT:  return static_cast<double>(xt.ftag);
    case FSDB_XTAG_TYPE_DOUBLE: return xt.dtag;
    case FSDB_XTAG_TYPE_L:      return static_cast<double>(xt.ltag.L);
    case FSDB_XTAG_TYPE_HL:
    default:                    return static_cast<double>((static_cast<uint64_t>(xt.hltag.H) << 32) | xt.hltag.L);
    }
}

static fsdbXTag double_to_xtag(double t, fsdbXTagType xtype) {
    fsdbXTag xt;
    std::memset(&xt, 0, sizeof(xt));
    switch (xtype) {
    case FSDB_XTAG_TYPE_FLOAT:
        xt.ftag = static_cast<float>(t);
        break;
    case FSDB_XTAG_TYPE_DOUBLE:
        xt.dtag = t;
        break;
    case FSDB_XTAG_TYPE_L:
        xt.ltag.L = static_cast<uint_T>(t);
        break;
    case FSDB_XTAG_TYPE_HL:
    default: {
        uint64_t v = static_cast<uint64_t>(t);
        xt.hltag.H = static_cast<uint_T>(v >> 32);
        xt.hltag.L = static_cast<uint_T>(v & 0xffffffffu);
        break;
    }
    }
    return xt;
}

static std::vector<Event> read_events(ffrObject *fsdb_obj, const Signal &sig, double end,
                                       fsdbXTagType xtype, bool verbose) {
    if (verbose) {
        std::cerr << "[read_events] signal=" << sig.full_name
                  << " idcode=" << sig.idcode
                  << " vc_dt=" << static_cast<int>(sig.vc_dt)
                  << " dtidcode=" << sig.dtidcode
                  << " type=" << static_cast<int>(sig.type)
                  << " bytes_per_bit=" << static_cast<int>(sig.bytes_per_bit)
                  << " xtag_type=";
        switch (xtype) {
        case FSDB_XTAG_TYPE_L:  std::cerr << "L\n"; break;
        case FSDB_XTAG_TYPE_HL: std::cerr << "HL\n"; break;
        case FSDB_XTAG_TYPE_FLOAT: std::cerr << "FLOAT\n"; break;
        case FSDB_XTAG_TYPE_DOUBLE: std::cerr << "DOUBLE\n"; break;
        default: std::cerr << static_cast<int>(xtype) << "\n"; break;
        }
    }
    std::vector<Event> events;
    fsdb_obj->ffrAddToSignalList(sig.idcode);
    fsdbRC load_rc = fsdb_obj->ffrLoadSignals();
    if (verbose) {
        std::cerr << "[read_events] ffrLoadSignals rc=" << static_cast<int>(load_rc)
                  << " file_type=" << static_cast<int>(fsdb_obj->ffrGetFileType())
                  << " dumper=" << (fsdb_obj->ffrGetFileDumperType() ? fsdb_obj->ffrGetFileDumperType() : "null")
                  << " sim=" << (fsdb_obj->ffrGetFileSimType() ? fsdb_obj->ffrGetFileSimType() : "null")
                  << "\n";
    }

    ffrVCTrvsHdl h = fsdb_obj->ffrCreateVCTrvsHdl(sig.idcode);
    if (!h) die("cannot create VC traverse handle for power signal: " + sig.full_name);
    if (verbose) std::cerr << "[read_events] VC traverse handle created\n";

    // Try ffrGotoTheFirstVC first (no XTag needed).
    // For EPIC/power data, ffrGotoXTag with time=0 may fail
    // because the FSDB's internal XTag format differs.
    fsdbRC rc = h->ffrGotoTheFirstVC();
    if (FSDB_RC_SUCCESS != rc) {
        // Fallback: try ffrGotoXTag with time 0 in the correct XTag format
        if (verbose) std::cerr << "[read_events] ffrGotoTheFirstVC failed (rc=" << rc
                               << "), trying ffrGotoXTag...\n";
        fsdbXTag xt_start = double_to_xtag(0.0, xtype);
        rc = h->ffrGotoXTag(&xt_start);
    }
    if (FSDB_RC_SUCCESS != rc) {
        // Last resort: try time-based VC traversal (for AMS/analog/EPIC data)
        h->ffrFree();
        fsdb_obj->ffrUnloadSignals();
        fsdb_obj->ffrResetSignalList();
        if (verbose) std::cerr << "[read_events] standard VC failed, trying time-based VC traversal...\n";
        fsdbVarIdcode ids[1] = { sig.idcode };
        ffrTimeBasedVCTrvsHdl tbh = fsdb_obj->ffrCreateTimeBasedVCTrvsHdl(1, ids);
        if (tbh) {
            // Time-based iterator may not need GotoXTag — try direct VC fetch
            if (verbose) std::cerr << "[read_events] time-based handle created, trying direct VC fetch...\n";
            {
                // Time-based traversal works — collect VCs using time-based handle
                size_t raw_count2 = 0, nan_count2 = 0, ok_count2 = 0;
                while (true) {
                    byte_T *vc_ptr = nullptr;
                    if (FSDB_RC_SUCCESS != tbh->ffrGetVC(&vc_ptr)) break;
                    raw_count2++;
                    fsdbXTag xt_cur2;
                    if (FSDB_RC_SUCCESS != tbh->ffrGetXTag(&xt_cur2)) break;
                    double t2 = xtag_to_double(xt_cur2, xtype);
                    double v2 = decode_numeric(sig.vc_dt, vc_ptr, sig.dtidcode);
                    if (verbose && raw_count2 <= 5) {
                        std::cerr << "[read_events] TB-VC#" << raw_count2 << " t=" << t2 << " raw=[";
                        for (int b = 0; b < 8; ++b)
                            std::cerr << " " << std::hex << static_cast<int>(vc_ptr[b]) << std::dec;
                        std::cerr << " ] val=" << v2 << "\n";
                    }
                    if (std::isfinite(v2)) { ok_count2++; events.push_back(Event{t2, v2}); }
                    else nan_count2++;
                    if (t2 > end + 1e-9) break;
                    if (FSDB_RC_SUCCESS != tbh->ffrGotoNextVC()) break;
                }
                if (verbose) std::cerr << "[read_events] time-based: raw=" << raw_count2
                                       << " ok=" << ok_count2 << " nan=" << nan_count2 << "\n";
                if (!events.empty()) {
                    tbh->ffrFree();
                    std::sort(events.begin(), events.end(),
                        [](const Event &a, const Event &b) { return a.time < b.time; });
                    return events;
                }
            }
            tbh->ffrFree();
        }
        fsdb_obj->ffrResetSignalList();
        die("no numeric value changes found for power signal: " + sig.full_name
            + " (standard VC, time-based VC all failed, standard rc="
            + std::to_string(static_cast<int>(rc)) + ")");
    }

    size_t raw_count = 0, nan_count = 0, ok_count = 0;
    fsdbXTag xt_cur;
    while (true) {
        byte_T *vc_ptr = nullptr;
        if (FSDB_RC_SUCCESS != h->ffrGetVC(&vc_ptr)) break;
        raw_count++;
        // Get time for current VC via GetXTag
        if (FSDB_RC_SUCCESS != h->ffrGetXTag(&xt_cur)) break;
        double t = xtag_to_double(xt_cur, xtype);
        double v = decode_numeric(sig.vc_dt, vc_ptr, sig.dtidcode);
        if (verbose && raw_count <= 5) {
            std::cerr << "[read_events] VC#" << raw_count << " t=" << t << " raw=[";
            for (int b = 0; b < 8; ++b)
                std::cerr << " " << std::hex << static_cast<int>(vc_ptr[b]) << std::dec;
            std::cerr << " ] val=" << v << "\n";
        }
        if (!std::isfinite(v)) {
            nan_count++;
            if (verbose && raw_count <= 5)
                std::cerr << "[read_events]   -> decode_numeric returned NaN/inf, skipping\n";
        } else {
            ok_count++;
            events.push_back(Event{t, v});
        }
        if (t > end + 1e-9) {
            if (verbose) std::cerr << "[read_events] t=" << t << " > end=" << end << ", stopping\n";
            break;
        }
        if (FSDB_RC_SUCCESS != h->ffrGotoNextVC()) break;
    }
    h->ffrFree();
    fsdb_obj->ffrUnloadSignals();
    fsdb_obj->ffrResetSignalList();

    if (verbose) {
        std::cerr << "[read_events] total_raw=" << raw_count
                  << " ok=" << ok_count << " nan=" << nan_count << "\n";
    }
    if (events.empty()) {
        if (raw_count > 0 && ok_count == 0) {
            die("all " + std::to_string(raw_count) + " value changes decoded to NaN for power signal: "
                + sig.full_name + " (vc_dt=" + std::to_string(static_cast<int>(sig.vc_dt))
                + " dtidcode=" + std::to_string(sig.dtidcode)
                + "). Run with --verbose to see raw bytes.");
        }
        die("no numeric value changes found for power signal: " + sig.full_name
            + " (raw_count=" + std::to_string(raw_count) + ")");
    }
    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) { return a.time < b.time; });
    return events;
}

static size_t nearest_event_index(const std::vector<Event> &events, double t) {
    auto it = std::lower_bound(events.begin(), events.end(), t,
        [](const Event &e, double value) { return e.time < value; });
    if (it == events.begin()) return 0;
    if (it == events.end()) return events.size() - 1;
    size_t hi = static_cast<size_t>(it - events.begin());
    size_t lo = hi - 1;
    double d_lo = std::abs(t - events[lo].time);
    double d_hi = std::abs(t - events[hi].time);
    return (d_lo <= d_hi) ? lo : hi;
}

static void usage(std::ostream &os) {
    os << "Usage: fsdb_to_power_label --fsdb <power.fsdb> --power-signal <name> --output-tsv <file> --step <tag> [--start <tag>] [--end <tag>] [options]\n\n"
       << "Options:\n"
       << "  --list-signals <file>  List numeric candidate signals and exit.\n"
       << "  --output-meta <file>   Optional JSON metadata output.\n"
       << "  omitted --start/--end  Use FSDB min/max simulation tags.\n"
       << "  --end-inclusive        Include end as a clock-edge sample.\n"
       << "  --verbose              More logs.\n";
}

static Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string &flag) -> std::string {
            if (i + 1 >= argc) die("missing value after " + flag);
            return argv[++i];
        };
        if (a == "--fsdb") opt.fsdb_path = need(a);
        else if (a == "--power-signal") opt.power_signal = need(a);
        else if (a == "--output-tsv") opt.output_tsv = need(a);
        else if (a == "--output-meta") opt.output_meta = need(a);
        else if (a == "--list-signals") opt.list_signals = need(a);
        else if (a == "--start") { opt.start = parse_double(need(a), a); opt.has_start = true; }
        else if (a == "--end") { opt.end = parse_double(need(a), a); opt.has_end = true; }
        else if (a == "--step") opt.step = parse_double(need(a), a);
        else if (a == "--end-inclusive") opt.end_inclusive = true;
        else if (a == "--verbose") opt.verbose = true;
        else if (a == "--help" || a == "-h") { usage(std::cout); std::exit(0); }
        else die("unknown argument: " + a);
    }
    if (opt.fsdb_path.empty()) die("--fsdb is required");
    if (opt.list_signals.empty()) {
        if (opt.power_signal.empty()) die("--power-signal is required");
        if (opt.output_tsv.empty()) die("--output-tsv is required");
        if (opt.step <= 0.0) die("--step must be > 0");
    }
    return opt;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);
        if (FALSE == ffrObject::ffrIsFSDB(const_cast<char *>(opt.fsdb_path.c_str()))) die(opt.fsdb_path + " is not an FSDB file");
        ffrObject *fsdb_obj = ffrObject::ffrOpen3(const_cast<char *>(opt.fsdb_path.c_str()));
        if (!fsdb_obj) die("ffrOpen3 failed: " + opt.fsdb_path);

        TreeContext ctx;
        const char *sep = fsdb_obj->ffrGetScopeSeparator();
        if (sep && *sep) ctx.sep = sep;

        // When listing signals, collect ALL signals so user can diagnose
        // which type PrimeTime PX used for power data
        if (!opt.list_signals.empty()) {
            ctx.list_all_signals = true;
        }

        fsdb_obj->ffrSetTreeCBFunc(tree_cb, &ctx);
        fsdb_obj->ffrReadScopeVarTree();

        if (!opt.list_signals.empty()) {
            mkdir_p(dirname_of(opt.list_signals));
            std::ofstream out(opt.list_signals);
            if (!out) die("cannot write signal list: " + opt.list_signals);
            for (const auto &s : ctx.signals) {
                out << s.full_name << " idcode=" << s.idcode << " vc_dt=" << static_cast<int>(s.vc_dt)
                    << " dtidcode=" << s.dtidcode << " type=" << static_cast<int>(s.type) << "\n";
            }
            fsdb_obj->ffrClose();
            std::cerr << "[fsdb_to_power_label] wrote signal list (all types): " << opt.list_signals
                      << " count=" << ctx.signals.size() << "\n";
            return 0;
        }

        apply_auto_time_range(fsdb_obj, opt);
        const Signal *sig = select_signal(ctx.signals, opt.power_signal);
        if (!sig) die("power signal not found: " + opt.power_signal);
        if (opt.verbose) {
            std::cerr << "[main] selected signal: " << sig->full_name
                      << " vc_dt=" << static_cast<int>(sig->vc_dt)
                      << " dtidcode=" << sig->dtidcode << "\n";
        }
        std::vector<double> sample_times = build_sample_times(opt);
        if (sample_times.empty()) die("empty sample time sequence");
        std::vector<Event> events = read_events(fsdb_obj, *sig, opt.end, opt.xtag_type, opt.verbose);

        mkdir_p(dirname_of(opt.output_tsv));
        std::ofstream out(opt.output_tsv);
        if (!out) die("cannot write output TSV: " + opt.output_tsv);
        out << "row\tclock_time\tpower_time\tdistance\tpower\n";
        double max_distance = 0.0;
        double sum_distance = 0.0;
        for (size_t row = 0; row < sample_times.size(); ++row) {
            double t = sample_times[row];
            size_t idx = nearest_event_index(events, t);
            const Event &e = events[idx];
            double d = std::abs(t - e.time);
            max_distance = std::max(max_distance, d);
            sum_distance += d;
            out << row << '\t' << t << '\t' << e.time << '\t' << d << '\t' << e.value << '\n';
        }
        if (!out) die("failed while writing output TSV: " + opt.output_tsv);

        if (!opt.output_meta.empty()) {
            mkdir_p(dirname_of(opt.output_meta));
            std::ofstream meta(opt.output_meta);
            if (!meta) die("cannot write output meta: " + opt.output_meta);
            meta << "{\n"
                 << "  \"fsdb\": \"" << json_escape(opt.fsdb_path) << "\",\n"
                 << "  \"power_signal\": \"" << json_escape(sig->full_name) << "\",\n"
                 << "  \"start\": " << opt.start << ",\n"
                 << "  \"end\": " << opt.end << ",\n"
                 << "  \"step\": " << opt.step << ",\n"
                 << "  \"end_inclusive\": " << (opt.end_inclusive ? "true" : "false") << ",\n"
                 << "  \"sample_rows\": " << sample_times.size() << ",\n"
                 << "  \"power_event_count\": " << events.size() << ",\n"
                 << "  \"max_distance\": " << max_distance << ",\n"
                 << "  \"mean_distance\": " << (sum_distance / static_cast<double>(sample_times.size())) << ",\n"
                 << "  \"output_tsv\": \"" << json_escape(opt.output_tsv) << "\"\n"
                 << "}\n";
        }

        fsdb_obj->ffrClose();
        std::cerr << "[fsdb_to_power_label] saved samples=" << sample_times.size()
                  << " events=" << events.size() << " signal=" << sig->full_name
                  << " tsv=" << opt.output_tsv << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
