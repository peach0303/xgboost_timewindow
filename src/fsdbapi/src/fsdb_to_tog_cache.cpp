#ifdef NOVAS_FSDB
#undef NOVAS_FSDB
#endif

#include "ffrAPI.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
    std::string output_dir;
    std::string dataset_name;
    std::string signal_list;
    std::string list_signals;
    std::string list_blocks;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t step = 0;
    bool has_start = false;
    bool has_end = false;
    size_t raw_col_block = 20000;
    size_t only_block_id = 0;  // 0 = export all blocks; block ids are 1-based.
    std::string tog_mode = "sample-xor";
    bool end_inclusive = false;
    bool drop_first_tog_row = false;
    bool write_level_cache = true;  // Writes one initial level row; full level is restored from initial + TOG.
    bool verbose = false;
    size_t max_cycles_per_file = 0;  // 0 = no limit (auto via ZIP64 safety)
    std::string debug_signal;
    size_t debug_row_start = 0;
    size_t debug_rows = 50;
    bool reverse_bus_bits = false;
    bool has_view_start = false;
    uint64_t view_start = 0;
};

struct Signal {
    fsdbVarIdcode idcode = 0;
    std::string full_name;
    std::string range_name;
    int lbit = 0;
    int rbit = 0;
    int bit_size = 1;
    byte_T bytes_per_bit = 0;
    byte_T type = 0;
};

struct BitColumn {
    size_t signal_index = 0;
    int bit_offset = 0;
    int bit_number = 0;
    std::string name;
    size_t raw_index = 0;  // 1-based, like CSV raw column index after time.
};

struct BlockInfo {
    size_t block_id = 0;
    size_t signal_begin = 0;
    size_t signal_end = 0;
    std::vector<BitColumn> bits;
};

struct TreeContext {
    std::vector<std::string> scope_stack;
    std::string sep = ".";
    std::vector<Signal> signals;
};

static std::string json_escape(const std::string &s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
        case '"': os << "\\\""; break;
        case '\\': os << "\\\\"; break;
        case '\b': os << "\\b"; break;
        case '\f': os << "\\f"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                os << buf;
            } else {
                os << static_cast<char>(c);
            }
        }
    }
    return os.str();
}

static void die(const std::string &msg) {
    throw std::runtime_error(msg);
}

static bool starts_with(const std::string &s, const std::string &prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

static std::string dirname_of(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static std::string basename_of(const std::string &path) {
    const size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static std::string stem_of(const std::string &path) {
    std::string base = basename_of(path);
    const size_t pos = base.find_last_of('.');
    return pos == std::string::npos ? base : base.substr(0, pos);
}

static std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

static void mkdir_p(const std::string &path) {
    if (path.empty() || path == ".") return;
    std::string cur;
    size_t i = 0;
    if (path[0] == '/') {
        cur = "/";
        i = 1;
    }
    while (i <= path.size()) {
        size_t slash = path.find('/', i);
        std::string part = path.substr(i, slash == std::string::npos ? std::string::npos : slash - i);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') cur += "/";
            cur += part;
            if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
                die("mkdir failed for " + cur + ": " + std::strerror(errno));
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
}

static std::string strip_range_suffix(const std::string &name) {
    const size_t rb = name.rfind(']');
    const size_t lb = name.rfind('[');
    if (rb == name.size() - 1 && lb != std::string::npos && lb < rb) {
        return name.substr(0, lb);
    }
    return name;
}

static bool has_range_suffix(const std::string &name) {
    const size_t rb = name.rfind(']');
    const size_t lb = name.rfind('[');
    return rb == name.size() - 1 && lb != std::string::npos && lb < rb;
}

static std::string make_range_name(const std::string &full_name, int lbit, int rbit, int bit_size) {
    if (bit_size <= 1 || has_range_suffix(full_name)) return full_name;
    return full_name + "[" + std::to_string(lbit) + ":" + std::to_string(rbit) + "]";
}

static std::string make_bit_name(const std::string &full_name, int lbit, int rbit, int bit_offset, int bit_size) {
    if (bit_size <= 1) return full_name;
    const int step = (lbit >= rbit) ? -1 : 1;
    const int bitnum = lbit + step * bit_offset;
    return strip_range_suffix(full_name) + "[" + std::to_string(bitnum) + "]";
}

static int bit_number_at(const Signal &sig, int bit_offset) {
    const int step = (sig.lbit >= sig.rbit) ? -1 : 1;
    return sig.lbit + step * bit_offset;
}

static bool supported_var(const fsdbTreeCBDataVar *var) {
    if (!var || var->is_dummy_var || var->is_void) return false;
    if (var->bytes_per_bit != FSDB_BYTES_PER_BIT_1B) return false;
    switch (var->type) {
    case FSDB_VT_VCD_REG:
    case FSDB_VT_VCD_WIRE:
    case FSDB_VT_VCD_TRI:
    case FSDB_VT_VCD_TRIAND:
    case FSDB_VT_VCD_TRIOR:
    case FSDB_VT_VCD_TRIREG:
    case FSDB_VT_VCD_TRI0:
    case FSDB_VT_VCD_TRI1:
    case FSDB_VT_VCD_WAND:
    case FSDB_VT_VCD_WOR:
    case FSDB_VT_VCD_INTEGER:
    case FSDB_VT_VCD_PARAMETER:
    case FSDB_VT_VHDL_SIGNAL:
    case FSDB_VT_VHDL_VARIABLE:
    case FSDB_VT_VHDL_CONSTANT:
        return true;
    default:
        return false;
    }
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
        if (!supported_var(var)) break;
        std::string scope = current_scope(*ctx);
        std::string local = var->name ? var->name : "";
        std::string full = scope.empty() ? local : scope + ctx->sep + local;
        Signal sig;
        sig.idcode = var->u.idcode;
        sig.full_name = full;
        sig.lbit = var->lbitnum;
        sig.rbit = var->rbitnum;
        sig.bit_size = std::abs(sig.lbit - sig.rbit) + 1;
        sig.bytes_per_bit = var->bytes_per_bit;
        sig.type = var->type;
        sig.range_name = make_range_name(full, sig.lbit, sig.rbit, sig.bit_size);
        ctx->signals.push_back(sig);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

static std::set<std::string> read_signal_filters(const std::string &path) {
    std::set<std::string> filters;
    if (path.empty()) return filters;
    std::ifstream in(path);
    if (!in) die("cannot open signal list: " + path);
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const size_t last = line.find_last_not_of(" \t\r\n");
        filters.insert(line.substr(first, last - first + 1));
    }
    return filters;
}

static bool signal_matches_filters(const Signal &sig, const std::set<std::string> &filters) {
    if (filters.empty()) return true;
    if (filters.count(sig.full_name) || filters.count(sig.range_name)) return true;
    for (const std::string &f : filters) {
        if (ends_with(f, "*")) {
            std::string prefix = f.substr(0, f.size() - 1);
            if (starts_with(sig.full_name, prefix) || starts_with(sig.range_name, prefix)) return true;
        }
    }
    for (int i = 0; i < sig.bit_size; ++i) {
        if (filters.count(make_bit_name(sig.full_name, sig.lbit, sig.rbit, i, sig.bit_size))) return true;
    }
    return false;
}

static uint64_t tag64_to_u64(const fsdbTag64 &t) {
    return (static_cast<uint64_t>(t.H) << 32) | static_cast<uint64_t>(t.L);
}

static uint64_t xtag_to_u64(const fsdbXTag &t) {
    return (static_cast<uint64_t>(t.hltag.H) << 32) | static_cast<uint64_t>(t.hltag.L);
}

static fsdbTag64 u64_to_tag64(uint64_t v) {
    fsdbTag64 t;
    t.H = static_cast<uint_T>(v >> 32);
    t.L = static_cast<uint_T>(v & 0xffffffffu);
    return t;
}

static fsdbXTag u64_to_xtag(uint64_t v) {
    fsdbXTag t;
    std::memset(&t, 0, sizeof(t));
    t.hltag.H = static_cast<uint_T>(v >> 32);
    t.hltag.L = static_cast<uint_T>(v & 0xffffffffu);
    return t;
}


static void apply_auto_time_range(ffrObject *fsdb_obj, Options &opt) {
    if (opt.has_start && opt.has_end) return;

    fsdbTag64 min_tag;
    fsdbTag64 max_tag;
    std::memset(&min_tag, 0, sizeof(min_tag));
    std::memset(&max_tag, 0, sizeof(max_tag));

    if (!opt.has_start) {
        if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMinFsdbTag64(&min_tag)) {
            die("cannot auto-detect FSDB min time tag; pass --start explicitly");
        }
        opt.start = tag64_to_u64(min_tag);
        opt.has_start = true;
    }
    if (!opt.has_end) {
        if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMaxFsdbTag64(&max_tag)) {
            die("cannot auto-detect FSDB max time tag; pass --end explicitly");
        }
        opt.end = tag64_to_u64(max_tag);
        opt.has_end = true;
    }
    if (opt.end <= opt.start) die("auto-detected --end must be larger than --start (start=" + std::to_string(opt.start) + ", end=" + std::to_string(opt.end) + ")");
}

static bool get_fsdb_time_range(ffrObject *fsdb_obj, uint64_t &min_time, uint64_t &max_time) {
    fsdbTag64 min_tag;
    fsdbTag64 max_tag;
    std::memset(&min_tag, 0, sizeof(min_tag));
    std::memset(&max_tag, 0, sizeof(max_tag));
    if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMinFsdbTag64(&min_tag)) return false;
    if (FSDB_RC_SUCCESS != fsdb_obj->ffrGetMaxFsdbTag64(&max_tag)) return false;
    min_time = tag64_to_u64(min_tag);
    max_time = tag64_to_u64(max_tag);
    return true;
}

static void warn_if_requested_time_looks_wrong(const Options &opt, uint64_t fsdb_min, uint64_t fsdb_max) {
    std::cerr << "[fsdb_to_tog_cache] fsdb_time_range min=" << fsdb_min
              << " max=" << fsdb_max << std::endl;
    if (opt.start < fsdb_min || opt.start > fsdb_max || opt.end < fsdb_min || opt.end > fsdb_max + 1) {
        std::cerr << "[fsdb_to_tog_cache][WARN] requested sample window looks outside FSDB time range: "
                  << "start=" << opt.start << " end=" << opt.end
                  << " fsdb_min=" << fsdb_min << " fsdb_max=" << fsdb_max
                  << ". If this is an fsdbextract output, its time axis may have been rebased; "
                  << "use the extracted FSDB's displayed/tag time coordinates." << std::endl;
    }
    if (opt.start > fsdb_min && opt.view_start >= opt.start) {
        std::cerr << "[fsdb_to_tog_cache][WARN] view_start equals/after sample start while FSDB has earlier data. "
                  << "Signals whose last assignment is before start may initialize incorrectly. "
                  << "Try --view-start " << fsdb_min << " or an earlier warmup tag." << std::endl;
    }
}

static std::vector<uint64_t> build_sample_times(const Options &opt) {
    if (opt.step == 0) die("--step must be > 0");
    if (opt.end <= opt.start) die("--end must be larger than --start (start=" + std::to_string(opt.start) + ", end=" + std::to_string(opt.end) + ")");
    std::vector<uint64_t> times;
    for (uint64_t t = opt.start; opt.end_inclusive ? (t <= opt.end) : (t < opt.end); ) {
        times.push_back(t);
        if (std::numeric_limits<uint64_t>::max() - t < opt.step) break;
        t += opt.step;
    }
    if (times.empty()) die("empty sample range");
    if (opt.drop_first_tog_row && times.size() < 2) {
        die("--drop-first-tog-row requires at least two sample points");
    }
    return times;
}

static void write_text_file(const std::string &path, const std::string &content) {
    mkdir_p(dirname_of(path));
    std::ofstream out(path);
    if (!out) die("cannot write: " + path);
    out << content;
}

static void write_binary_file(const std::string &path, const std::vector<uint8_t> &content) {
    mkdir_p(dirname_of(path));
    std::ofstream out(path, std::ios::binary);
    if (!out) die("cannot write: " + path);
    out.write(reinterpret_cast<const char *>(content.data()), static_cast<std::streamsize>(content.size()));
    if (!out) die("failed while writing: " + path);
}

static void write_lines_file(const std::string &path, const std::vector<std::string> &lines) {
    mkdir_p(dirname_of(path));
    std::ofstream out(path);
    if (!out) die("cannot write: " + path);
    for (const auto &line : lines) out << line << "\n";
}

// -------------------------
// Minimal .npy/.npz writer.
// This writes uncompressed ZIP entries. numpy.load/scipy.load_npz accept this.
// Standard ZIP cannot handle individual entries >= 4 GiB without ZIP64.
// We use a conservative threshold (4 GiB - 64 KiB) for safety.
// -------------------------

static constexpr size_t kZipEntrySafeBytes = 0xFFFF0000u;  // ~4 GiB minus 64 KiB

static uint32_t crc32_update(uint32_t crc, const uint8_t *buf, size_t len) {
    static uint32_t table[256];
    static bool have_table = false;
    if (!have_table) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t rem = i;
            for (int j = 0; j < 8; ++j)
                rem = (rem >> 1) ^ (0xedb88320u & (0u - (rem & 1u)));
            table[i] = rem;
        }
        have_table = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < len; ++i)
        crc = table[(crc ^ buf[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

static void append_u16(std::vector<uint8_t> &v, uint16_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xff));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xff));
}

static void append_u32(std::vector<uint8_t> &v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xff));
}

static void append_bytes(std::vector<uint8_t> &v, const void *data, size_t len) {
    const auto *p = static_cast<const uint8_t *>(data);
    v.insert(v.end(), p, p + len);
}

static std::string shape_repr(const std::vector<size_t> &shape) {
    if (shape.empty()) return "()";
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ", ";
        os << shape[i];
    }
    if (shape.size() == 1) os << ",";
    os << ")";
    return os.str();
}

static std::vector<uint8_t> make_npy(const std::string &descr,
                                     const std::vector<size_t> &shape,
                                     const void *data,
                                     size_t data_bytes) {
    std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                         shape_repr(shape) + ", }";
    const size_t preamble = 10;
    size_t pad = 16 - ((preamble + header.size() + 1) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');
    if (header.size() > 65535) die("NPY header too large");
    std::vector<uint8_t> out;
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    append_bytes(out, magic, sizeof(magic));
    out.push_back(1);
    out.push_back(0);
    append_u16(out, static_cast<uint16_t>(header.size()));
    append_bytes(out, header.data(), header.size());
    append_bytes(out, data, data_bytes);
    return out;
}

struct ZipEntry {
    std::string name;
    std::vector<uint8_t> data;
    uint32_t crc = 0;
    uint32_t local_offset = 0;
};

static void write_npz(const std::string &path,
                      const std::vector<uint8_t> &data_npy,
                      const std::vector<uint8_t> &indices_npy,
                      const std::vector<uint8_t> &indptr_npy,
                      const std::vector<uint8_t> &shape_npy,
                      const std::vector<uint8_t> &format_npy) {
    mkdir_p(dirname_of(path));
    std::vector<ZipEntry> entries = {
        {"data.npy", data_npy, 0, 0},
        {"indices.npy", indices_npy, 0, 0},
        {"indptr.npy", indptr_npy, 0, 0},
        {"shape.npy", shape_npy, 0, 0},
        {"format.npy", format_npy, 0, 0},
    };

    std::vector<uint8_t> file;
    for (auto &e : entries) {
        if (e.data.size() > 0xffffffffu) die("ZIP64 is not implemented; entry too large: " + e.name);
        if (file.size() > 0xffffffffu) die("ZIP64 is not implemented; npz too large");
        e.crc = crc32_update(0, e.data.data(), e.data.size());
        e.local_offset = static_cast<uint32_t>(file.size());
        append_u32(file, 0x04034b50u);
        append_u16(file, 20);
        append_u16(file, 0);
        append_u16(file, 0);  // store
        append_u16(file, 0);
        append_u16(file, 33); // 1980-01-01 in DOS date format.
        append_u32(file, e.crc);
        append_u32(file, static_cast<uint32_t>(e.data.size()));
        append_u32(file, static_cast<uint32_t>(e.data.size()));
        append_u16(file, static_cast<uint16_t>(e.name.size()));
        append_u16(file, 0);
        append_bytes(file, e.name.data(), e.name.size());
        append_bytes(file, e.data.data(), e.data.size());
    }

    const uint32_t cd_offset = static_cast<uint32_t>(file.size());
    for (const auto &e : entries) {
        append_u32(file, 0x02014b50u);
        append_u16(file, 20);
        append_u16(file, 20);
        append_u16(file, 0);
        append_u16(file, 0);
        append_u16(file, 0);
        append_u16(file, 33);
        append_u32(file, e.crc);
        append_u32(file, static_cast<uint32_t>(e.data.size()));
        append_u32(file, static_cast<uint32_t>(e.data.size()));
        append_u16(file, static_cast<uint16_t>(e.name.size()));
        append_u16(file, 0);
        append_u16(file, 0);
        append_u16(file, 0);
        append_u16(file, 0);
        append_u32(file, 0);
        append_u32(file, e.local_offset);
        append_bytes(file, e.name.data(), e.name.size());
    }
    const uint32_t cd_size = static_cast<uint32_t>(file.size() - cd_offset);
    append_u32(file, 0x06054b50u);
    append_u16(file, 0);
    append_u16(file, 0);
    append_u16(file, static_cast<uint16_t>(entries.size()));
    append_u16(file, static_cast<uint16_t>(entries.size()));
    append_u32(file, cd_size);
    append_u32(file, cd_offset);
    append_u16(file, 0);

    std::ofstream out(path, std::ios::binary);
    if (!out) die("cannot write npz: " + path);
    out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
    if (!out) die("failed while writing npz: " + path);
}

static void write_csc_npz(const std::string &path,
                          size_t n_rows,
                          size_t n_cols,
                          const std::vector<uint8_t> &data,
                          const std::vector<int32_t> &indices,
                          const std::vector<int64_t> &indptr) {
    if (indptr.size() != n_cols + 1) die("bad CSC indptr length");
    if (data.size() != indices.size()) die("bad CSC data/indices length");
    int64_t shape_arr[2] = {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)};
    const char fmt[3] = {'c', 's', 'c'};
    auto data_npy = make_npy("|u1", {data.size()}, data.data(), data.size());
    auto indices_npy = make_npy("<i4", {indices.size()}, indices.data(), indices.size() * sizeof(int32_t));
    auto indptr_npy = make_npy("<i8", {indptr.size()}, indptr.data(), indptr.size() * sizeof(int64_t));
    auto shape_npy = make_npy("<i8", {2}, shape_arr, sizeof(shape_arr));
    auto format_npy = make_npy("|S3", {}, fmt, sizeof(fmt));
    write_npz(path, data_npy, indices_npy, indptr_npy, shape_npy, format_npy);
}

enum class BitDecode {
    Zero,
    One,
    X,
    Z,
    NoChange,
    NoValue,
    Other,
};

static BitDecode decode_bit(byte_T b, uint8_t &val) {
    switch (b) {
    case FSDB_BT_VCD_0:
        val = 0;
        return BitDecode::Zero;
    case FSDB_BT_VCD_1:
        val = 1;
        return BitDecode::One;
    case FSDB_BT_VCD_X:
        val = 0;
        return BitDecode::X;
    case FSDB_BT_VCD_Z:
        val = 0;
        return BitDecode::Z;
    case FSDB_BT_VCD_NC:
        val = 0;
        return BitDecode::NoChange;
    case FSDB_BT_VCD_NV:
        val = 0;
        return BitDecode::NoValue;
    default:
        val = 0;
        return BitDecode::Other;
    }
}

static char decode_bit_char(byte_T b) {
    uint8_t val = 0;
    switch (decode_bit(b, val)) {
    case BitDecode::Zero: return '0';
    case BitDecode::One: return '1';
    case BitDecode::X: return 'x';
    case BitDecode::Z: return 'z';
    case BitDecode::NoChange: return 'n';
    case BitDecode::NoValue: return 'v';
    case BitDecode::Other: return '?';
    }
    return '?';
}

static std::string vc_to_string(byte_T *vc_ptr, int bit_size) {
    if (!vc_ptr) return "<null>";
    std::string s;
    s.reserve(static_cast<size_t>(bit_size));
    for (int i = 0; i < bit_size; ++i) s.push_back(decode_bit_char(vc_ptr[i]));
    return s;
}

struct BlockResult {
    std::vector<std::string> npz_paths;       // 1 entry = single file; >1 = row-sharded
    std::string initial_level_path;            // one uint8 .npy vector for level row 0
    std::vector<std::string> clean_names;
    std::vector<std::string> removed_xz_names;
    size_t nnz = 0;
    size_t initial_level_nnz = 0;
    size_t n_rows = 0;
    size_t level_n_rows = 0;
};

static void mark_touched_col(size_t col,
                             std::vector<size_t> *touched_cols,
                             std::vector<uint8_t> *touched_flag) {
    if (!touched_cols || !touched_flag) return;
    if (!(*touched_flag)[col]) {
        (*touched_flag)[col] = 1;
        touched_cols->push_back(col);
    }
}

static void apply_vc_to_state(const std::vector<Signal> &block_signals,
                              const std::unordered_map<fsdbVarIdcode, std::vector<size_t>> &local_signal_by_id,
                              const std::vector<size_t> &bit_base_by_signal,
                              fsdbVarIdcode idcode,
                              byte_T *vc_ptr,
                              std::vector<uint8_t> &current,
                              std::vector<uint8_t> &valid,
                              std::vector<uint8_t> &changed,
                              std::vector<uint8_t> &xz_event_seen,
                              std::vector<uint8_t> &nc_event_seen,
                              std::vector<uint8_t> &nv_event_seen,
                              std::vector<uint8_t> &other_event_seen,
                              bool mark_changed,
                              bool reverse_bus_bits,
                              std::vector<size_t> *touched_cols = nullptr,
                              std::vector<uint8_t> *touched_flag = nullptr) {
    auto it = local_signal_by_id.find(idcode);
    if (it == local_signal_by_id.end()) return;
    for (size_t sidx : it->second) {
        const Signal &sig = block_signals[sidx];
        const size_t base = bit_base_by_signal[sidx];
        for (int bit = 0; bit < sig.bit_size; ++bit) {
            const size_t col = base + static_cast<size_t>(bit);
            const int vc_bit = reverse_bus_bits ? (sig.bit_size - 1 - bit) : bit;
            uint8_t new_val = 0;
            const BitDecode decoded = decode_bit(vc_ptr[vc_bit], new_val);
            if (decoded == BitDecode::NoChange) {
                nc_event_seen[col] = 1;
                continue;
            }
            if (decoded == BitDecode::NoValue) {
                nv_event_seen[col] = 1;
                continue;
            }
            if (decoded == BitDecode::Other) {
                other_event_seen[col] = 1;
                continue;
            }
            if (decoded == BitDecode::X || decoded == BitDecode::Z) {
                xz_event_seen[col] = 1;
                // Only true FSDB X/Z can make the sampled state invalid. NC/NV are
                // Reader/internal states and are handled as hold/default-zero.
                mark_touched_col(col, touched_cols, touched_flag);
                valid[col] = 0;
                current[col] = 0;
                continue;
            }
            const uint8_t old_valid = valid[col];
            const uint8_t old_val = current[col];
            if (mark_changed && valid[col] && current[col] != new_val) {
                changed[col] = 1;
            }
            current[col] = new_val;
            valid[col] = 1;
            if (!old_valid || old_val != new_val) {
                mark_touched_col(col, touched_cols, touched_flag);
            }
        }
    }
}

static BlockResult export_block(ffrObject *fsdb_obj,
                                const Options &opt,
                                const std::vector<Signal> &all_signals,
                                const BlockInfo &block,
                                const std::vector<uint64_t> &times,
                                const std::string &cache_dir) {
    BlockResult result;
    const size_t sample_rows = times.size();
    const size_t output_rows = opt.drop_first_tog_row ? sample_rows - 1 : sample_rows;
    const size_t row_offset = opt.drop_first_tog_row ? 1 : 0;
    const size_t n_cols = block.bits.size();

    std::vector<Signal> block_signals;
    block_signals.reserve(block.signal_end - block.signal_begin);
    for (size_t i = block.signal_begin; i < block.signal_end; ++i) {
        block_signals.push_back(all_signals[i]);
    }

    std::vector<size_t> bit_base_by_signal(block_signals.size() + 1, 0);
    for (size_t i = 0; i < block_signals.size(); ++i) {
        bit_base_by_signal[i + 1] = bit_base_by_signal[i] + static_cast<size_t>(block_signals[i].bit_size);
    }
    if (bit_base_by_signal.back() != n_cols) die("internal bit count mismatch");

    size_t debug_col = std::numeric_limits<size_t>::max();
    fsdbVarIdcode debug_idcode = 0;
    size_t debug_local_signal = std::numeric_limits<size_t>::max();
    if (!opt.debug_signal.empty()) {
        std::vector<size_t> exact_matches;
        std::vector<size_t> substring_matches;
        for (size_t i = 0; i < block.bits.size(); ++i) {
            if (block.bits[i].name == opt.debug_signal) exact_matches.push_back(i);
            if (block.bits[i].name.find(opt.debug_signal) != std::string::npos) substring_matches.push_back(i);
        }
        if (exact_matches.size() == 1) {
            debug_col = exact_matches[0];
        } else if (exact_matches.empty() && substring_matches.size() == 1) {
            debug_col = substring_matches[0];
        } else if (!exact_matches.empty() || substring_matches.size() > 1) {
            std::cerr << "[fsdb_to_tog_cache][debug] ambiguous debug signal in block " << block.block_id
                      << ": " << opt.debug_signal << " matches="
                      << (exact_matches.empty() ? substring_matches.size() : exact_matches.size()) << std::endl;
        }
        if (debug_col != std::numeric_limits<size_t>::max()) {
            debug_local_signal = block.bits[debug_col].signal_index - block.signal_begin;
            debug_idcode = block_signals[debug_local_signal].idcode;
            std::cerr << "[fsdb_to_tog_cache][debug] block " << block.block_id
                      << " debug_signal=" << block.bits[debug_col].name
                      << " local_col=" << debug_col
                      << " bit_pos=" << block.bits[debug_col].bit_offset
                      << " reverse_bus_bits=" << (opt.reverse_bus_bits ? "true" : "false")
                      << " idcode=" << debug_idcode
                      << " vc_bit=" << (opt.reverse_bus_bits ? (block_signals[debug_local_signal].bit_size - 1 - block.bits[debug_col].bit_offset) : block.bits[debug_col].bit_offset)
                      << " sample_rows=" << sample_rows << std::endl;
        }
    }

    std::unordered_map<fsdbVarIdcode, std::vector<size_t>> local_signal_by_id;
    std::vector<fsdbVarIdcode> sig_arr;
    std::set<fsdbVarIdcode> added_idcodes;
    size_t alias_idcode_count = 0;
    for (size_t i = 0; i < block_signals.size(); ++i) {
        auto &indices = local_signal_by_id[block_signals[i].idcode];
        if (!indices.empty()) ++alias_idcode_count;
        indices.push_back(i);
        if (added_idcodes.insert(block_signals[i].idcode).second) {
            sig_arr.push_back(block_signals[i].idcode);
            fsdb_obj->ffrAddToSignalList(block_signals[i].idcode);
        }
    }
    if (alias_idcode_count > 0) {
        std::cerr << "[fsdb_to_tog_cache][WARN] block " << block.block_id
                  << " has " << alias_idcode_count
                  << " signals sharing an FSDB idcode with an earlier signal; events will be applied to all aliases"
                  << std::endl;
    }

    std::cerr << "[fsdb_to_tog_cache] load block " << block.block_id
              << " signals=" << block_signals.size()
              << " bits=" << n_cols << std::endl;

    fsdb_obj->ffrLoadSignals();
    std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
              << " signals loaded; initialize state" << std::endl;

    std::vector<uint8_t> current(n_cols, 0);
    // Match fsdbreport -of b -bit behavior for no-value initial state: default to 0.
    std::vector<uint8_t> valid(n_cols, 1);
    std::vector<uint8_t> prev_sample(n_cols, 0);
    std::vector<uint8_t> prev_valid(n_cols, 0);
    std::vector<uint8_t> changed(n_cols, 0);
    std::vector<uint8_t> dirty(n_cols, 0);
    std::vector<uint8_t> xz_event_seen(n_cols, 0);
    std::vector<uint8_t> nc_event_seen(n_cols, 0);
    std::vector<uint8_t> nv_event_seen(n_cols, 0);
    std::vector<uint8_t> other_event_seen(n_cols, 0);

    const size_t init_progress_stride = std::max<size_t>(1, block_signals.size() / 20);
    size_t init_handles = 0;
    size_t init_incore = 0;
    size_t init_goto_ok = 0;
    size_t init_getvc_ok = 0;
    size_t init_after_start = 0;
    for (size_t sidx = 0; sidx < block_signals.size(); ++sidx) {
        if (sidx == 0 || sidx + 1 == block_signals.size() || (sidx % init_progress_stride) == 0) {
            std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                      << " init signal " << sidx << "/" << block_signals.size()
                      << " handles=" << init_handles
                      << " incore=" << init_incore
                      << " goto_ok=" << init_goto_ok
                      << " getvc_ok=" << init_getvc_ok
                      << std::endl;
        }
        ffrVCTrvsHdl h = fsdb_obj->ffrCreateVCTrvsHdl(block_signals[sidx].idcode);
        if (!h) continue;
        ++init_handles;
        if (h->ffrHasIncoreVC()) {
            ++init_incore;
            fsdbTag64 q = u64_to_tag64(opt.start);
            if (FSDB_RC_SUCCESS == h->ffrGotoXTag(&q)) {
                ++init_goto_ok;
                const uint64_t aligned_t = tag64_to_u64(q);
                if (aligned_t > opt.start) ++init_after_start;
                byte_T *vc_ptr = nullptr;
                if (aligned_t <= opt.start && FSDB_RC_SUCCESS == h->ffrGetVC(&vc_ptr)) {
                    ++init_getvc_ok;
                    if (block_signals[sidx].idcode == debug_idcode) {
                        std::cerr << "[fsdb_to_tog_cache][debug] init_vc query_start=" << opt.start
                                  << " aligned_t=" << aligned_t
                                  << " idcode=" << block_signals[sidx].idcode
                                  << " raw_vc=" << vc_to_string(vc_ptr, block_signals[sidx].bit_size)
                                  << " reverse_bus_bits=" << (opt.reverse_bus_bits ? "true" : "false")
                                  << std::endl;
                    }
                    apply_vc_to_state(block_signals, local_signal_by_id, bit_base_by_signal,
                                      block_signals[sidx].idcode, vc_ptr,
                                      current, valid, changed, xz_event_seen, nc_event_seen, nv_event_seen, other_event_seen, false, opt.reverse_bus_bits);
                }
            }
        }
        h->ffrFree();
    }
    std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
              << " init done signals=" << block_signals.size()
              << " handles=" << init_handles
              << " incore=" << init_incore
              << " goto_ok=" << init_goto_ok
              << " getvc_ok=" << init_getvc_ok
              << " aligned_after_start=" << init_after_start
              << std::endl;
    if (init_after_start > 0 || init_getvc_ok < init_handles) {
        std::cerr << "[fsdb_to_tog_cache][WARN] block " << block.block_id
                  << " initial-state coverage may be incomplete: handles=" << init_handles
                  << " getvc_ok=" << init_getvc_ok
                  << " aligned_after_start=" << init_after_start
                  << ". This often happens when the FSDB/view window starts after the last assignment. "
                  << "Use --view-start earlier, or extract the FSDB with initial values at the cut point."
                  << std::endl;
    }

    ffrTimeBasedVCTrvsHdl tb = fsdb_obj->ffrCreateTimeBasedVCTrvsHdl(
        static_cast<uint_T>(sig_arr.size()), sig_arr.data());

    bool has_event = false;
    fsdbVarIdcode event_id = 0;
    fsdbXTag event_time;
    fsdbSeqNum event_seq = 0;
    byte_T *event_vc = nullptr;

    auto read_current_event = [&]() -> bool {
        if (!tb) return false;
        if (FSDB_RC_SUCCESS == tb->ffrGetVarIdcodeXTagVCSeqNum(&event_id, &event_time, &event_vc, &event_seq)) {
            return true;
        }
        return false;
    };
    auto advance_event = [&]() -> bool {
        if (!tb) return false;
        if (FSDB_RC_SUCCESS != tb->ffrGotoNextVC()) return false;
        return read_current_event();
    };

    has_event = advance_event();
    std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
              << " time traversal ready; sample_rows=" << sample_rows
              << " output_rows=" << output_rows << std::endl;

    std::vector<std::vector<int32_t>> rows_by_col(n_cols);
    std::vector<uint8_t> initial_level_by_col(n_cols, 0);
    const bool any_change_mode = (opt.tog_mode == "any-change");

    const size_t progress_stride = std::max<size_t>(1, sample_rows / 20);
    size_t event_count = 0;
    std::vector<size_t> touched_cols;
    std::vector<uint8_t> touched_flag(n_cols, 0);
    touched_cols.reserve(std::min<size_t>(n_cols, 1024));

    for (size_t row = 0; row < sample_rows; ++row) {
        if (row == 0 || row + 1 == sample_rows || (row % progress_stride) == 0) {
            std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                      << " progress row=" << row << "/" << sample_rows
                      << " events=" << event_count
                      << " touched=" << touched_cols.size() << std::endl;
        }
        const uint64_t sample_t = times[row];
        const uint64_t prev_t = (row == 0) ? sample_t : times[row - 1];

        while (has_event && xtag_to_u64(event_time) <= sample_t) {
            const uint64_t et = xtag_to_u64(event_time);
            const bool mark_changed = any_change_mode && row > 0 && et > prev_t;
            const bool debug_this_event = (event_id == debug_idcode &&
                                           row >= opt.debug_row_start &&
                                           row < opt.debug_row_start + opt.debug_rows &&
                                           debug_col != std::numeric_limits<size_t>::max() &&
                                           debug_local_signal != std::numeric_limits<size_t>::max());
            uint8_t debug_before_current = 0;
            uint8_t debug_before_valid = 0;
            int debug_vc_bit = -1;
            char debug_vc_char = '?';
            if (debug_this_event) {
                const Signal &dbg_sig = block_signals[debug_local_signal];
                debug_before_current = current[debug_col];
                debug_before_valid = valid[debug_col];
                debug_vc_bit = opt.reverse_bus_bits
                    ? (dbg_sig.bit_size - 1 - block.bits[debug_col].bit_offset)
                    : block.bits[debug_col].bit_offset;
                if (event_vc && debug_vc_bit >= 0 && debug_vc_bit < dbg_sig.bit_size) {
                    debug_vc_char = decode_bit_char(event_vc[debug_vc_bit]);
                }
                std::cerr << "[fsdb_to_tog_cache][debug] event_before row=" << row
                          << " sample_t=" << sample_t
                          << " event_t=" << et
                          << " seq=" << event_seq
                          << " raw_vc=" << vc_to_string(event_vc, dbg_sig.bit_size)
                          << " target_vc_bit=" << debug_vc_bit
                          << " target_vc=" << debug_vc_char
                          << " before_current=" << static_cast<int>(debug_before_current)
                          << " before_valid=" << static_cast<int>(debug_before_valid)
                          << std::endl;
            }
            apply_vc_to_state(block_signals, local_signal_by_id, bit_base_by_signal,
                              event_id, event_vc, current, valid, changed, xz_event_seen, nc_event_seen, nv_event_seen, other_event_seen, mark_changed, opt.reverse_bus_bits,
                              &touched_cols, &touched_flag);
            if (debug_this_event) {
                std::cerr << "[fsdb_to_tog_cache][debug] event_after row=" << row
                          << " sample_t=" << sample_t
                          << " event_t=" << et
                          << " target_vc=" << debug_vc_char
                          << " after_current=" << static_cast<int>(current[debug_col])
                          << " after_valid=" << static_cast<int>(valid[debug_col])
                          << " touched=" << static_cast<int>(touched_flag[debug_col])
                          << " changed=" << static_cast<int>(changed[debug_col])
                          << std::endl;
            }
            ++event_count;
            has_event = advance_event();
        }

        if (row == 0) {
            for (size_t col = 0; col < n_cols; ++col) {
                if (!valid[col]) dirty[col] = 1;
                if (opt.write_level_cache) {
                    initial_level_by_col[col] = (valid[col] && current[col]) ? 1 : 0;
                }
            }
            if (debug_col != std::numeric_limits<size_t>::max() &&
                row >= opt.debug_row_start && row < opt.debug_row_start + opt.debug_rows) {
                std::cerr << "[fsdb_to_tog_cache][debug] row=" << row
                          << " sample_t=" << sample_t
                          << " init_current=" << static_cast<int>(current[debug_col])
                          << " valid=" << static_cast<int>(valid[debug_col])
                          << " touched=" << static_cast<int>(touched_flag[debug_col])
                          << std::endl;
            }
            prev_sample = current;
            prev_valid = valid;
        } else {
            if (any_change_mode) {
                for (size_t col : touched_cols) {
                    uint8_t tog = 0;
                    if (!valid[col]) {
                        dirty[col] = 1;
                    } else {
                        tog = changed[col] ? 1 : 0;
                    }
                    if (tog && row >= row_offset) {
                        rows_by_col[col].push_back(static_cast<int32_t>(row - row_offset));
                    }
                }
            } else {
                for (size_t col = 0; col < n_cols; ++col) {
                    uint8_t tog = 0;
                    if (!valid[col]) {
                        dirty[col] = 1;
                    } else if (!prev_valid[col]) {
                        dirty[col] = 1;
                    } else {
                        tog = (current[col] != prev_sample[col]) ? 1 : 0;
                    }
                    if (tog && row >= row_offset) {
                        rows_by_col[col].push_back(static_cast<int32_t>(row - row_offset));
                    }
                }
            }

            if (debug_col != std::numeric_limits<size_t>::max() &&
                row >= opt.debug_row_start && row < opt.debug_row_start + opt.debug_rows) {
                uint8_t debug_tog = 0;
                if (valid[debug_col] && prev_valid[debug_col]) {
                    debug_tog = any_change_mode ? changed[debug_col]
                                                : (current[debug_col] != prev_sample[debug_col] ? 1 : 0);
                }
                std::cerr << "[fsdb_to_tog_cache][debug] row=" << row
                          << " sample_t=" << sample_t
                          << " prev_t=" << prev_t
                          << " prev=" << static_cast<int>(prev_sample[debug_col])
                          << " current=" << static_cast<int>(current[debug_col])
                          << " prev_valid=" << static_cast<int>(prev_valid[debug_col])
                          << " valid=" << static_cast<int>(valid[debug_col])
                          << " touched=" << static_cast<int>(touched_flag[debug_col])
                          << " changed=" << static_cast<int>(changed[debug_col])
                          << " computed_tog=" << static_cast<int>(debug_tog)
                          << " output_row=" << (row >= row_offset ? static_cast<long long>(row - row_offset) : -1)
                          << std::endl;
            }

            if (any_change_mode) {
                for (size_t col : touched_cols) {
                    prev_sample[col] = current[col];
                    prev_valid[col] = valid[col];
                    changed[col] = 0;
                }
            } else {
                prev_sample = current;
                prev_valid = valid;
            }
        }

        for (size_t col : touched_cols) touched_flag[col] = 0;
        touched_cols.clear();
    }

    std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
              << " sample loop done; events=" << event_count
              << "; writing sparse outputs" << std::endl;

    if (tb) tb->ffrFree();
    fsdb_obj->ffrUnloadSignals();
    fsdb_obj->ffrResetSignalList();

    std::vector<uint8_t> data;
    std::vector<int32_t> indices;
    std::vector<int64_t> indptr;
    indptr.reserve(n_cols + 1);
    indptr.push_back(0);
    std::vector<uint8_t> initial_level_clean;
    initial_level_clean.reserve(n_cols);

    std::vector<size_t> clean_col_indices;  // original col indices that passed XZ check
    size_t true_xz_event_cols = 0;
    size_t transient_only_xz_cols = 0;
    size_t nc_event_cols = 0;
    size_t nv_event_cols = 0;
    size_t other_event_cols = 0;
    for (size_t col = 0; col < n_cols; ++col) {
        if (xz_event_seen[col]) {
            ++true_xz_event_cols;
            if (!dirty[col]) ++transient_only_xz_cols;
        }
        if (nc_event_seen[col]) ++nc_event_cols;
        if (nv_event_seen[col]) ++nv_event_cols;
        if (other_event_seen[col]) ++other_event_cols;
        if (dirty[col]) {
            result.removed_xz_names.push_back(block.bits[col].name);
            continue;
        }
        result.clean_names.push_back(block.bits[col].name);
        clean_col_indices.push_back(col);
        const auto &rows = rows_by_col[col];
        for (int32_t r : rows) {
            indices.push_back(r);
            data.push_back(1);
        }
        indptr.push_back(static_cast<int64_t>(indices.size()));
        if (opt.write_level_cache) {
            const uint8_t v = initial_level_by_col[col] ? 1 : 0;
            initial_level_clean.push_back(v);
            if (v) ++result.initial_level_nnz;
        }
    }

    result.nnz = indices.size();
    // Full level rows are intentionally not stored. They are restored later as
    // initial_level XOR cumulative TOG after TOG-based column cleaning.

    // Helper: write one CSC shard (or single file) to an NPZ path.
    auto save_one_shard = [&](const std::string &path,
                               size_t shard_n_rows, size_t shard_n_cols,
                               const std::vector<uint8_t> &shard_data,
                               const std::vector<int32_t> &shard_idx,
                               const std::vector<int64_t> &shard_ptr) {
        if (shard_idx.empty()) {
            // Edge case — write empty matrix with correct shape.
            std::vector<uint8_t> empty_data;
            std::vector<int32_t> empty_idx;
            std::vector<int64_t> empty_ptr(static_cast<size_t>(shard_n_cols) + 1, 0);
            write_csc_npz(path, shard_n_rows, shard_n_cols, empty_data, empty_idx, empty_ptr);
        } else {
            write_csc_npz(path, shard_n_rows, shard_n_cols, shard_data, shard_idx, shard_ptr);
        }
    };

    // Helper: estimate total .npz file size (all entries + headers).
    // We need the TOTAL to stay under the ZIP 4 GiB limit, not just the
    // largest single entry, because write_npz also checks the overall file size.
    auto total_npz_bytes = [&](size_t nnz_count, size_t n_cols_count) -> size_t {
        const size_t data_sz = nnz_count;                                // data.npy
        const size_t idx_sz  = nnz_count * sizeof(int32_t);             // indices.npy
        const size_t ptr_sz  = (n_cols_count + 1) * sizeof(int64_t);    // indptr.npy
        const size_t shape_sz = 2 * sizeof(int64_t) + 128;              // shape.npy
        const size_t fmt_sz = 3 + 128;                                  // format.npy
        // NPY headers (128 B each) + ZIP local & central dir overhead (~200 B per entry)
        const size_t overhead = 5 * 400;
        return data_sz + idx_sz + ptr_sz + shape_sz + fmt_sz + overhead;
    };

    // --------------- TOG save (single or row-sharded) ---------------
    {
        const size_t n_clean = result.clean_names.size();
        const size_t full_npz_bytes = total_npz_bytes(result.nnz, n_clean);

        const bool user_cycle_limit = (opt.max_cycles_per_file > 0 && output_rows > opt.max_cycles_per_file);
        const bool need_shard = (full_npz_bytes > kZipEntrySafeBytes) || user_cycle_limit;

        std::ostringstream fn;
        fn << opt.dataset_name << ".block";
        fn.width(5); fn.fill('0');
        fn << block.block_id << ".tog.cache.csc";
        const std::string base_path = join_path(cache_dir, fn.str());

        if (!need_shard) {
            // Single file
            const std::string path = base_path + ".npz";
            save_one_shard(path, output_rows, n_clean, data, indices, indptr);
            result.npz_paths.push_back(path);
            result.n_rows = output_rows;
        } else {
            // Compute ZIP64-safe max rows per shard
            const double bytes_per_nnz = static_cast<double>(sizeof(int32_t) + 1);
            double nnz_per_row = static_cast<double>(result.nnz) / std::max(1.0, static_cast<double>(output_rows));
            size_t max_rows_per_shard = std::max(static_cast<size_t>(1),
                static_cast<size_t>(static_cast<double>(kZipEntrySafeBytes) /
                                    std::max(1e-9, nnz_per_row * bytes_per_nnz)));
            // User-specified cycle limit takes precedence
            if (opt.max_cycles_per_file > 0) {
                max_rows_per_shard = std::min(max_rows_per_shard, opt.max_cycles_per_file);
            }
            max_rows_per_shard = std::min(max_rows_per_shard, output_rows);

            if (user_cycle_limit && full_npz_bytes <= kZipEntrySafeBytes) {
                std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                          << " TOG row-sharding by --max-cycles-per-file=" << opt.max_cycles_per_file
                          << " (output_rows=" << output_rows << ")\n";
            }

            size_t total_rows_written = 0;
            for (size_t r0 = 0, si = 0; r0 < output_rows; r0 += max_rows_per_shard, ++si) {
                size_t r1 = std::min(output_rows, r0 + max_rows_per_shard);
                std::ostringstream sp;
                sp << base_path << ".row" << si << ".npz";
                const std::string shard_path = sp.str();

                std::vector<uint8_t> sd;
                std::vector<int32_t> si_idx;
                std::vector<int64_t> si_ptr = {0};
                for (size_t ci : clean_col_indices) {
                    const auto &rvec = rows_by_col[ci];
                    for (int32_t r : rvec) {
                        if (static_cast<size_t>(r) >= r0 && static_cast<size_t>(r) < r1) {
                            si_idx.push_back(static_cast<int32_t>(static_cast<size_t>(r) - r0));
                            sd.push_back(1);
                        }
                    }
                    si_ptr.push_back(static_cast<int64_t>(si_idx.size()));
                }

                save_one_shard(shard_path, r1 - r0, n_clean, sd, si_idx, si_ptr);
                result.npz_paths.push_back(shard_path);
                total_rows_written += (r1 - r0);

                // Log each shard
                std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                          << " TOG shard " << si << " rows=" << (r1 - r0) << " nnz=" << si_idx.size()
                          << " path=" << shard_path << "\n";
            }
            result.n_rows = total_rows_written;
            std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                      << " TOG row-sharded: " << result.npz_paths.size() << " shards"
                      << " full_npz=" << (full_npz_bytes >> 20) << " MiB > safe=" << (kZipEntrySafeBytes >> 20) << " MiB\n";
        }
    }

    // --------------- Initial level save ---------------
    if (opt.write_level_cache) {
        std::ostringstream fn;
        fn << opt.dataset_name << ".block";
        fn.width(5); fn.fill('0');
        fn << block.block_id << ".initial_level.cache.uint8.npy";
        const std::string path = join_path(cache_dir, fn.str());
        auto initial_npy = make_npy("|u1", {initial_level_clean.size()},
                                    initial_level_clean.data(), initial_level_clean.size());
        write_binary_file(path, initial_npy);
        result.initial_level_path = path;
        result.level_n_rows = sample_rows;
        std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
                  << " initial LEVEL saved: cols=" << initial_level_clean.size()
                  << " nnz=" << result.initial_level_nnz
                  << " path=" << path << "\n";
    }

    std::cerr << "[fsdb_to_tog_cache] block " << block.block_id
              << " saved: clean_cols=" << result.clean_names.size()
              << " removed_xz=" << result.removed_xz_names.size()
              << " true_xz_event_cols=" << true_xz_event_cols
              << " transient_only_xz=" << transient_only_xz_cols
              << " nc_event_cols=" << nc_event_cols
              << " nv_event_cols=" << nv_event_cols
              << " other_event_cols=" << other_event_cols
              << " nnz=" << result.nnz
              << " initial_level_nnz=" << result.initial_level_nnz
              << " n_paths=" << result.npz_paths.size();
    if (opt.write_level_cache) std::cerr << " initial_level=" << (result.initial_level_path.empty() ? 0 : 1);
    std::cerr << "\n";

    return result;
}

static std::vector<BlockInfo> build_blocks(const std::vector<Signal> &signals, size_t raw_col_block) {
    std::vector<BlockInfo> blocks;
    size_t block_id = 1;
    for (size_t begin = 0; begin < signals.size(); begin += raw_col_block) {
        size_t end = std::min(signals.size(), begin + raw_col_block);
        BlockInfo b;
        b.block_id = block_id++;
        b.signal_begin = begin;
        b.signal_end = end;
        for (size_t s = begin; s < end; ++s) {
            const Signal &sig = signals[s];
            for (int bit = 0; bit < sig.bit_size; ++bit) {
                BitColumn col;
                col.signal_index = s;
                col.bit_offset = bit;
                col.bit_number = bit_number_at(sig, bit);
                col.raw_index = s + 1;
                col.name = make_bit_name(sig.full_name, sig.lbit, sig.rbit, bit, sig.bit_size);
                b.bits.push_back(col);
            }
        }
        blocks.push_back(std::move(b));
    }
    return blocks;
}

static void write_block_list_tsv(const std::string &path, const std::vector<BlockInfo> &blocks) {
    mkdir_p(dirname_of(path));
    std::ofstream out(path);
    if (!out) die("cannot write block list: " + path);
    out << "block_id\tsignal_begin\tsignal_end\tsignals\tbits\n";
    for (const auto &b : blocks) {
        out << b.block_id << "\t"
            << b.signal_begin << "\t"
            << b.signal_end << "\t"
            << (b.signal_end - b.signal_begin) << "\t"
            << b.bits.size() << "\n";
    }
}

static void write_layout_jsonl(const std::string &path,
                               const std::vector<Signal> &signals,
                               const std::vector<BlockInfo> &blocks) {
    mkdir_p(dirname_of(path));
    std::ofstream out(path);
    if (!out) die("cannot write layout: " + path);
    size_t expanded_idx = 0;
    for (const auto &b : blocks) {
        for (const auto &col : b.bits) {
            const Signal &sig = signals[col.signal_index];
            out << "{\"expanded_idx\":" << expanded_idx++
                << ",\"name\":\"" << json_escape(col.name)
                << "\",\"raw_idx\":" << col.raw_index
                << ",\"raw_name\":\"" << json_escape(sig.range_name)
                << "\",\"width\":" << sig.bit_size
                << ",\"bit_pos\":" << col.bit_offset
                << "}\n";
        }
    }
}

static std::vector<std::string> build_original_headers(const std::vector<Signal> &signals) {
    std::vector<std::string> headers;
    headers.push_back("time");
    for (const auto &s : signals) headers.push_back(s.range_name);
    return headers;
}

static void write_manifest(const std::string &manifest_path,
                           const Options &opt,
                           const std::string &csv_key,
                           const std::vector<Signal> &signals,
                           const std::vector<BlockInfo> &blocks,
                           const std::vector<BlockResult> &results,
                           size_t sample_rows,
                           size_t output_rows) {
    mkdir_p(dirname_of(manifest_path));
    std::ofstream out(manifest_path);
    if (!out) die("cannot write manifest: " + manifest_path);
    out << "{\n";
    out << "  \"cache_format\": \"csc\",\n";
    out << "  \"input_data_type\": \"tog\",\n";
    out << "  \"tog_diff_applied\": false,\n";
    out << "  \"tog_note\": \"Generated directly from FSDB levels by fsdbapi; no CSV was produced.\",\n";
    out << "  \"level_cache_available\": " << (opt.write_level_cache ? "true" : "false") << ",\n";
    out << "  \"level_cache_mode\": \"initial_plus_tog\",\n";
    out << "  \"level_note\": \"Only initial sampled level row is stored as uint8 .npy; full level is restored from initial level plus TOG after column cleaning.\",\n";
    out << "  \"xz_policy\": \"sampled-level-only true X/Z dirty; NC holds previous; NV/no-value defaults to 0 for fsdbreport bit CSV compatibility\",\n";
    out << "  \"source_fsdb\": \"" << json_escape(opt.fsdb_path) << "\",\n";
    out << "  \"sample_start\": " << opt.start << ",\n";
    out << "  \"sample_end\": " << opt.end << ",\n";
    out << "  \"sample_step\": " << opt.step << ",\n";
    out << "  \"sample_end_inclusive\": " << (opt.end_inclusive ? "true" : "false") << ",\n";
    out << "  \"drop_first_tog_row\": " << (opt.drop_first_tog_row ? "true" : "false") << ",\n";
    out << "  \"sample_rows\": " << sample_rows << ",\n";
    out << "  \"level_rows\": " << sample_rows << ",\n";
    out << "  \"output_rows\": " << output_rows << ",\n";
    out << "  \"tog_mode\": \"" << json_escape(opt.tog_mode) << "\",\n";
    out << "  \"blocks\": [\n";
    for (size_t i = 0; i < blocks.size(); ++i) {
        const BlockInfo &b = blocks[i];
        const BlockResult &r = results[i];
        out << "    {\n";
        out << "      \"block_id\": " << b.block_id << ",\n";
        out << "      \"raw_indices\": [";
        for (size_t s = b.signal_begin; s < b.signal_end; ++s) {
            if (s != b.signal_begin) out << ", ";
            out << (s + 1);
        }
        out << "],\n";
        out << "      \"names\": [";
        for (size_t j = 0; j < r.clean_names.size(); ++j) {
            if (j) out << ", ";
            out << "\"" << json_escape(r.clean_names[j]) << "\"";
        }
        out << "],\n";
        out << "      \"removed_xz_names\": [";
        for (size_t j = 0; j < r.removed_xz_names.size(); ++j) {
            if (j) out << ", ";
            out << "\"" << json_escape(r.removed_xz_names[j]) << "\"";
        }
        out << "],\n";
        // files: single string or JSON array for row-sharded blocks
        out << "      \"files\": {\"" << json_escape(csv_key) << "\": ";
        if (r.npz_paths.size() == 1) {
            out << "\"" << json_escape(r.npz_paths[0]) << "\"";
        } else {
            out << "[";
            for (size_t p = 0; p < r.npz_paths.size(); ++p) {
                if (p) out << ", ";
                out << "\"" << json_escape(r.npz_paths[p]) << "\"";
            }
            out << "]";
        }
        out << "},\n";
        out << "      \"level_files\": {},\n";
        if (opt.write_level_cache) {
            out << "      \"initial_level_files\": {\"" << json_escape(csv_key) << "\": "
                << "\"" << json_escape(r.initial_level_path) << "\"},\n";
        } else {
            out << "      \"initial_level_files\": {},\n";
        }
        out << "      \"n_rows\": {\"" << json_escape(csv_key) << "\": " << r.n_rows << "},\n";
        if (opt.write_level_cache) {
            out << "      \"level_n_rows\": {\"" << json_escape(csv_key) << "\": " << r.level_n_rows << "},\n";
        } else {
            out << "      \"level_n_rows\": {},\n";
        }
        out << "      \"nnz\": {\"" << json_escape(csv_key) << "\": " << r.nnz << "},\n";
        if (opt.write_level_cache) {
            out << "      \"initial_level_nnz\": {\"" << json_escape(csv_key) << "\": " << r.initial_level_nnz << "}\n";
        } else {
            out << "      \"initial_level_nnz\": {}\n";
        }
        out << "    }" << (i + 1 == blocks.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"csv_paths\": [\"" << json_escape(csv_key) << "\"],\n";
    out << "  \"clean_names_after_xz\": [";
    bool first = true;
    size_t xz_removed = 0;
    for (const auto &r : results) {
        xz_removed += r.removed_xz_names.size();
        for (const auto &n : r.clean_names) {
            if (!first) out << ", ";
            first = false;
            out << "\"" << json_escape(n) << "\"";
        }
    }
    out << "],\n";
    out << "  \"xz_removed_count\": " << xz_removed << ",\n";
    out << "  \"num_blocks\": " << blocks.size() << ",\n";
    out << "  \"raw_signal_count\": " << signals.size() << "\n";
    out << "}\n";
}

static void usage(std::ostream &os) {
    os << "Usage: fsdb_to_tog_cache --fsdb <wave.fsdb> --output-dir <dir> "
          "--step <tag> [--start <tag>] [--end <tag>] [options]\n\n"
       << "Options:\n"
       << "  --dataset-name <name>       Stem used in cache file names (default: fsdb stem)\n"
       << "  --signal-list <file>        Optional signal filter. Lines are full bus names,\n"
       << "                              full bit names, or prefixes ending with '*'.\n"
       << "  --list-signals <file>       Only list supported FSDB signals and exit.\n"
       << "  --list-blocks <file>        List cache block partition after signal filtering and exit.\n"
       << "  --only-block-id <id>        Export only this 1-based cache block id.\n"
       << "  omitted --start/--end       Use FSDB min/max simulation tags.\n"
       << "  --raw-col-block <n>         Raw bus signals per cache block (default: 20000)\n"
       << "  --tog-mode <mode>           sample-xor or any-change (default: sample-xor)\n"
       << "  --end-inclusive             Include --end as a sample point.\n"
       << "  --drop-first-tog-row        Drop the first dummy TOG row after diff.\n"
       << "  --no-level-cache            Do not write initial level row cache.\n"
       << "  --max-cycles-per-file <n>   Max cycles per output NPZ (0=auto, ZIP64 only).\n"
       << "  --debug-signal <name>       Print sampled internal state for one expanded bit column.\n"
       << "  --debug-row-start <n>       First sample row to print for --debug-signal.\n"
       << "  --debug-rows <n>            Number of sample rows to print for --debug-signal.\n"
       << "  --reverse-bus-bits          Decode vc bytes in reverse order for multi-bit signals.\n"
       << "  --view-start <tag>          Load FSDB events from this tag instead of --start for correct initial state.\n"
       << "  --verbose                   More progress logs\n";
}

static uint64_t parse_u64(const std::string &s, const std::string &name) {
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s.c_str(), &end, 0);
    if (errno || !end || *end) die("bad integer for " + name + ": " + s);
    return static_cast<uint64_t>(v);
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
        else if (a == "--output-dir") opt.output_dir = need(a);
        else if (a == "--dataset-name") opt.dataset_name = need(a);
        else if (a == "--signal-list") opt.signal_list = need(a);
        else if (a == "--list-signals") opt.list_signals = need(a);
        else if (a == "--list-blocks") opt.list_blocks = need(a);
        else if (a == "--start") { opt.start = parse_u64(need(a), a); opt.has_start = true; }
        else if (a == "--end") { opt.end = parse_u64(need(a), a); opt.has_end = true; }
        else if (a == "--step") opt.step = parse_u64(need(a), a);
        else if (a == "--raw-col-block") opt.raw_col_block = static_cast<size_t>(parse_u64(need(a), a));
        else if (a == "--only-block-id") opt.only_block_id = static_cast<size_t>(parse_u64(need(a), a));
        else if (a == "--tog-mode") opt.tog_mode = need(a);
        else if (a == "--end-inclusive") opt.end_inclusive = true;
        else if (a == "--drop-first-tog-row") opt.drop_first_tog_row = true;
        else if (a == "--no-level-cache") opt.write_level_cache = false;
        else if (a == "--max-cycles-per-file") opt.max_cycles_per_file = static_cast<size_t>(parse_u64(need(a), a));
        else if (a == "--debug-signal") opt.debug_signal = need(a);
        else if (a == "--debug-row-start") opt.debug_row_start = static_cast<size_t>(parse_u64(need(a), a));
        else if (a == "--debug-rows") opt.debug_rows = static_cast<size_t>(parse_u64(need(a), a));
        else if (a == "--reverse-bus-bits") opt.reverse_bus_bits = true;
        else if (a == "--view-start") { opt.view_start = parse_u64(need(a), a); opt.has_view_start = true; }
        else if (a == "--verbose") opt.verbose = true;
        else if (a == "--help" || a == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            die("unknown argument: " + a);
        }
    }
    if (opt.fsdb_path.empty()) die("--fsdb is required");
    if (opt.output_dir.empty() && opt.list_signals.empty() && opt.list_blocks.empty()) die("--output-dir is required unless --list-signals/--list-blocks is used");
    if (opt.dataset_name.empty()) opt.dataset_name = stem_of(opt.fsdb_path);
    if (opt.raw_col_block == 0) die("--raw-col-block must be > 0");
    if (opt.tog_mode != "sample-xor" && opt.tog_mode != "any-change") {
        die("--tog-mode must be sample-xor or any-change");
    }
    if (opt.write_level_cache && opt.tog_mode != "sample-xor") {
        die("initial_plus_tog level restoration requires --tog-mode sample-xor; use --no-level-cache for any-change");
    }
    if (opt.list_signals.empty() && opt.list_blocks.empty()) {
        if (opt.step == 0) die("--step is required");
        if (opt.has_start && opt.has_end && opt.end <= opt.start) die("--end must be larger than --start (start=" + std::to_string(opt.start) + ", end=" + std::to_string(opt.end) + ")");
        if (opt.has_view_start && opt.has_start && opt.view_start > opt.start) {
            die("--view-start must be <= --start (view_start=" + std::to_string(opt.view_start) + ", start=" + std::to_string(opt.start) + ")");
        }
    }
    return opt;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);

        if (FALSE == ffrObject::ffrIsFSDB(const_cast<char *>(opt.fsdb_path.c_str()))) {
            die(opt.fsdb_path + " is not an FSDB file");
        }

        std::cerr << "[fsdb_to_tog_cache] opening fsdb=" << opt.fsdb_path << std::endl;
        ffrObject *fsdb_obj = ffrObject::ffrOpen3(const_cast<char *>(opt.fsdb_path.c_str()));
        if (!fsdb_obj) die("ffrOpen3 failed: " + opt.fsdb_path);
        std::cerr << "[fsdb_to_tog_cache] opened fsdb; reading scope/var tree" << std::endl;

        TreeContext ctx;
        const char *sep = fsdb_obj->ffrGetScopeSeparator();
        if (sep && *sep) ctx.sep = sep;
        fsdb_obj->ffrSetTreeCBFunc(tree_cb, &ctx);
        fsdb_obj->ffrReadScopeVarTree();
        std::cerr << "[fsdb_to_tog_cache] scope/var tree loaded; supported_signals="
                  << ctx.signals.size() << std::endl;

        if (!opt.list_signals.empty()) {
            mkdir_p(dirname_of(opt.list_signals));
            std::ofstream out(opt.list_signals);
            if (!out) die("cannot write signal list: " + opt.list_signals);
            for (const auto &s : ctx.signals) {
                out << s.range_name << " idcode=" << s.idcode << " bits=" << s.bit_size << "\n";
            }
            fsdb_obj->ffrClose();
            std::cerr << "[fsdb_to_tog_cache] wrote supported signal list: " << opt.list_signals
                      << " count=" << ctx.signals.size() << "\n";
            return 0;
        }

        std::cerr << "[fsdb_to_tog_cache] reading signal filters" << std::endl;
        std::set<std::string> filters = read_signal_filters(opt.signal_list);
        std::vector<Signal> signals;
        std::set<std::string> seen_names;
        for (const auto &s : ctx.signals) {
            if (signal_matches_filters(s, filters) && seen_names.insert(s.full_name).second) {
                signals.push_back(s);
            }
        }
        if (signals.empty()) die("no supported signals selected");
        std::cerr << "[fsdb_to_tog_cache] selected signals=" << signals.size()
                  << " from filters=" << (opt.signal_list.empty() ? "<none>" : opt.signal_list)
                  << std::endl;

        std::vector<BlockInfo> all_blocks = build_blocks(signals, opt.raw_col_block);
        if (!opt.list_blocks.empty()) {
            write_block_list_tsv(opt.list_blocks, all_blocks);
            fsdb_obj->ffrClose();
            std::cerr << "[fsdb_to_tog_cache] wrote block list: " << opt.list_blocks
                      << " blocks=" << all_blocks.size() << std::endl;
            return 0;
        }

        std::cerr << "[fsdb_to_tog_cache] applying time range" << std::endl;
        apply_auto_time_range(fsdb_obj, opt);
        if (!opt.has_view_start) opt.view_start = opt.start;
        if (opt.view_start > opt.start) {
            die("--view-start must be <= --start (view_start=" + std::to_string(opt.view_start) + ", start=" + std::to_string(opt.start) + ")");
        }
        uint64_t fsdb_min_time = 0;
        uint64_t fsdb_max_time = 0;
        if (get_fsdb_time_range(fsdb_obj, fsdb_min_time, fsdb_max_time)) {
            warn_if_requested_time_looks_wrong(opt, fsdb_min_time, fsdb_max_time);
        } else {
            std::cerr << "[fsdb_to_tog_cache][WARN] cannot query FSDB min/max time range" << std::endl;
        }

        const std::vector<uint64_t> times = build_sample_times(opt);
        const size_t output_rows = opt.drop_first_tog_row ? times.size() - 1 : times.size();
        mkdir_p(opt.output_dir);
        const std::string work_dir = join_path(opt.output_dir, "_work");
        const std::string cache_dir = join_path(work_dir, "tog_cache");
        const std::string marker_dir = join_path(work_dir, "source_markers");
        mkdir_p(cache_dir);
        mkdir_p(marker_dir);

        fsdbXTag vw_start = u64_to_xtag(opt.view_start);
        fsdbXTag vw_end = u64_to_xtag(opt.end_inclusive ? opt.end : (opt.end == 0 ? opt.end : opt.end - 1));
        fsdb_obj->ffrSetViewWindow(&vw_start, &vw_end);

        std::cerr << "[fsdb_to_tog_cache] fsdb=" << opt.fsdb_path
                  << " selected_signals=" << signals.size()
                  << " sample_rows=" << times.size()
                  << " output_rows=" << output_rows
                  << " start=" << opt.start
                  << " end=" << opt.end
                  << " view_start=" << opt.view_start
                  << " step=" << opt.step
                  << " end_inclusive=" << (opt.end_inclusive ? "true" : "false")
                  << " drop_first_tog_row=" << (opt.drop_first_tog_row ? "true" : "false")
                  << " level_cache=" << (opt.write_level_cache ? "true" : "false")
                  << " mode=" << opt.tog_mode
                  << " reverse_bus_bits=" << (opt.reverse_bus_bits ? "true" : "false")
                  << " xz_policy=sampled-level-only,NC-hold,NV-default0,true-XZ-only" << "\n";

        std::vector<BlockInfo> blocks;
        if (opt.only_block_id != 0) {
            for (const auto &b : all_blocks) {
                if (b.block_id == opt.only_block_id) {
                    blocks.push_back(b);
                    break;
                }
            }
            if (blocks.empty()) die("--only-block-id not found: " + std::to_string(opt.only_block_id));
        } else {
            blocks = all_blocks;
        }
        std::vector<BlockResult> results;
        results.reserve(blocks.size());
        for (const auto &b : blocks) {
            results.push_back(export_block(fsdb_obj, opt, signals, b, times, cache_dir));
        }

        const std::string csv_key = join_path(marker_dir, opt.dataset_name + ".fsdbtog");
        {
            std::ostringstream marker;
            marker << "source_fsdb=" << opt.fsdb_path << "\n"
                   << "dataset_name=" << opt.dataset_name << "\n"
                   << "sample_rows=" << times.size() << "\n"
                   << "rows=" << output_rows << "\n";
            write_text_file(csv_key, marker.str());
        }

        std::vector<std::string> headers = build_original_headers(signals);
        write_lines_file(join_path(opt.output_dir, "original_headers.txt"), headers);
        write_layout_jsonl(join_path(work_dir, "expanded_layout.jsonl"), signals, all_blocks);
        write_manifest(join_path(work_dir, "tog_cache_manifest.json"), opt, csv_key,
                       signals, blocks, results, times.size(), output_rows);

        fsdb_obj->ffrClose();
        std::cerr << "[fsdb_to_tog_cache] done. manifest="
                  << join_path(work_dir, "tog_cache_manifest.json") << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
