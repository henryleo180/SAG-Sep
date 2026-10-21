#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "csr.h"
#include "forward.h"
#include "io_vrp.h"
#include "json_out.h"
#include "loop.h"
#include "sagsep.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static int usage(FILE *out)
{
    fprintf(out,
        "usage:\n"
        "  sagsep --instance <name|path.vrp> --size-preset <nXXX> [options]\n"
        "  sagsep --config <ref.toml> --instance <name> [options]\n"
        "  sagsep --config <ref.toml> [options]        # run the whole [[instances]] list in one process\n"
        "  sagsep <command> [args]\n"
        "\n"
        "run options:\n"
        "  --instance <s>        a single instance name (instances_root/<name>.vrp) or a .vrp path;\n"
        "                        with --config it must appear in the TOML's [[instances]] list.\n"
        "                        If omitted, every instance runs in TOML order (the model loads once)\n"
        "  --size-preset <s>     frozen per-size preset (n50|n75|n100|n200|n300|n400|n500|n750|n1000)\n"
        "  --recipe <r>          with --size-preset: full (default) = Sweep + Onion + TTS, the released\n"
        "                        recipe; os = Onion-Sweep (Sweep + Onion, TTS off).\n"
        "                        With --config the TOML's [tts] enabled decides\n"
        "  --config <f.toml>     reference TOML (minimal reader: an unknown key is an error)\n"
        "  --vehicles <K>        RHS 2K of the depot degree constraint; if omitted, inferred as ceil(d(V)/Q)\n"
        "  --optimal <v>         known optimum (recorded in the JSON only)\n"
        "  --max-iter <N>        override the iteration limit\n"
        "  --time-limit <s>      override the wall-clock limit (<=0 = unlimited)\n"
        "  --instances-root <d>  override the instance root directory\n"
        "  --out <dir>           directory for the result JSON (root_relax_<stem>_learned_fb.json);\n"
        "                        with --config the TOML's output_dir is the fallback; an error if neither is given\n"
        "  --device <s>          forward execution provider: cpu (bit-for-bit reproducible) | cuda[:id]\n"
        "                        (much faster, not bit-for-bit reproducible). Defaults to the TOML's policy_device\n"
        "  --onnx <model.onnx>   run the forward pass through onnxruntime (needs ONNXRUNTIME_HOME at build time)\n"
        "  --no-diagnostics      omit the rcc_pipeline / tts diagnostics from the JSON (still collected)\n"
        "  --quiet               do not print per-iteration progress\n"
        "\n"
        "commands:\n"
        "  preset  <size>             print the frozen configuration\n"
        "  build                      print the build-time switches that were detected\n"
        "  vrp <file.vrp> <vehicles>  read a TSPLIB instance and print a summary\n"
        "\n"
        "Unimplemented paths always fail loudly: any decode/onion/tts value outside the frozen\n"
        "recipe is rejected during configuration validation rather than silently coerced.\n");
    return 2;
}

static int cmd_preset(int argc, char **argv)
{
    if (argc != 1) return usage(stderr);
    sag_config cfg = sag_config_preset(argv[0]);
    sag_config_print(stdout, &cfg);
    return 0;
}

static int cmd_build(void)
{
    printf("sagsep build configuration\n");
#ifdef HAVE_GUROBI
    printf("  gurobi        yes\n");
#else
    printf("  gurobi        no  (required: the build errors out when src/lp.c is present)\n");
#endif
#ifdef HAVE_ONNXRUNTIME
    printf("  onnxruntime   yes (the ONNX forward pass is available)\n");
#else
    printf("  onnxruntime   no  (the ONNX forward pass is NOT available)\n");
#endif
#ifdef _OPENMP
    printf("  openmp        yes (_OPENMP=%d, max_threads=%d) -- note: src/ contains no"
           " `#pragma omp` at all, so the cut families run serially\n",
           _OPENMP, omp_get_max_threads());
#else
    printf("  openmp        no\n");
#endif
    double nan_value = sag_quiet_nan();
    uint64_t nan_bits;
    memcpy(&nan_bits, &nan_value, sizeof nan_bits);
    printf("  quiet NaN     0x%016llx\n", (unsigned long long)nan_bits);
    return 0;
}

static int cmd_vrp(int argc, char **argv)
{
    if (argc != 2) return usage(stderr);
    char *end = NULL;
    long long vehicles = strtoll(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0') {
        fprintf(stderr, "vehicles must be an integer, got '%s'\n", argv[1]);
        return 2;
    }
    sag_arena a;
    sag_arena_init(&a, 128u << 20, "cli_vrp");
    sag_instance inst;
    sag_load_instance(&a, argv[0], (int64_t)vehicles, false, 0.0, &inst);
    printf("instance      %s\n", inst.name);
    printf("n_nodes       %lld  (depot=%lld)\n", (long long)inst.n_nodes, (long long)inst.depot);
    printf("capacity      %lld\n", (long long)inst.capacity);
    printf("vehicles      %lld  (K_depot)\n", (long long)inst.vehicles);
    printf("vehicle_count %lld  (K_replica; -1 means features.c fills it in)\n",
           (long long)inst.vehicle_count);
    int64_t total_demand = 0;
    for (int64_t i = 0; i < inst.n_nodes; ++i) total_demand += inst.demand[i];
    printf("demand_sum    %lld\n", (long long)total_demand);
    printf("coords[0]     %.17g %.17g\n", inst.coords[0], inst.coords[1]);
    printf("coords[N-1]   %.17g %.17g\n",
           inst.coords[(inst.n_nodes - 1) * 2], inst.coords[(inst.n_nodes - 1) * 2 + 1]);
    printf("weight[0,1]   %.17g\n", inst.weight[1]);
    printf("weight[1,0]   %.17g\n", inst.weight[inst.n_nodes]);
    double wsum = 0.0;
    for (int64_t i = 0; i < inst.n_nodes * inst.n_nodes; ++i) wsum += inst.weight[i];
    printf("weight_sum    %.17g\n", wsum);
    printf("arena_used    %zu bytes\n", sag_arena_used(&a));
    sag_arena_destroy(&a);
    return 0;
}

/* ------------------------------------------------------------------ */
/* run mode                                                            */
/* ------------------------------------------------------------------ */

static void ensure_dir(const char *dir)
{
    char buf[1024];
    const size_t n = strlen(dir);
    if (n + 1 > sizeof buf) sag_die("output directory path too long: %s", dir);
    memcpy(buf, dir, n + 1);
    for (size_t i = 1; i <= n; ++i) {
        if (buf[i] != '/' && buf[i] != '\0') continue;
        const char saved = buf[i];
        buf[i] = '\0';
        if (mkdir(buf, 0777) != 0) {
            struct stat st;
            if (stat(buf, &st) != 0 || !S_ISDIR(st.st_mode))
                sag_die("cannot create output directory %s", buf);
        }
        buf[i] = saved;
    }
}

static void resolve_vrp_path(char *out, size_t cap, const char *name, const char *root)
{
    const size_t len = strlen(name);
    int n;
    if (len >= 4 && strcmp(name + len - 4, ".vrp") == 0) n = snprintf(out, cap, "%s", name);
    else n = snprintf(out, cap, "%s/%s.vrp", root, name);
    if (n < 0 || (size_t)n >= cap) sag_die("instance path too long: %s", name);
}

static void instance_stem(char *out, size_t cap, const char *name)
{
    const char *base = strrchr(name, '/');
    base = base ? base + 1 : name;
    const size_t len = strlen(base);
    size_t keep = len;
    if (len >= 4 && strcmp(base + len - 4, ".vrp") == 0) keep = len - 4;
    if (keep + 1 > cap) sag_die("instance name too long: %s", base);
    memcpy(out, base, keep);
    out[keep] = '\0';
}

static int64_t infer_vehicles(const sag_instance *inst)
{
    double total = 0.0;
    for (int64_t i = 0; i < inst->n_nodes; ++i) total += (double)inst->demand[i];
    double cap = (double)inst->capacity;
    if (cap < 1e-6) cap = 1e-6;
    double k = ceil(total / cap);
    if (!(k >= 1.0)) k = 1.0;
    return (int64_t)k;
}

static long long parse_ll(const char *s, const char *what)
{
    char *end = NULL;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end != '\0') sag_die("%s must be an integer, got '%s'", what, s);
    return v;
}

static double parse_f64(const char *s, const char *what)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || *end != '\0') sag_die("%s must be a floating-point number, got '%s'", what, s);
    return v;
}

typedef struct {
    const char *name;
    int64_t     vehicles;
    int64_t     max_iter;
    double      optimal;
    bool        has_optimal;
} run_spec;

static int cmd_run(int argc, char **argv)
{
    const char *instance_arg = NULL, *size_preset = NULL, *toml_path = NULL;
    const char *out_dir = NULL, *onnx_path = NULL;
    const char *instances_root = NULL, *device = NULL, *recipe = NULL;
    int64_t vehicles = -1, max_iter = -1;
    double optimal = 0.0, time_limit = -1.0;
    bool has_optimal = false, has_time_limit = false;
    bool emit_diag = true, quiet = false;
    bool has_vehicles = false, has_max_iter = false;

    for (int i = 0; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDV(flag) do { if (!v) sag_die("%s needs an argument", flag); ++i; } while (0)
        if (strcmp(a, "--instance") == 0)            { NEEDV(a); instance_arg = v; }
        else if (strcmp(a, "--size-preset") == 0)    { NEEDV(a); size_preset = v; }
        else if (strcmp(a, "--config") == 0)         { NEEDV(a); toml_path = v; }
        else if (strcmp(a, "--recipe") == 0)         { NEEDV(a); recipe = v; }
        else if (strcmp(a, "--out") == 0)            { NEEDV(a); out_dir = v; }
        else if (strcmp(a, "--onnx") == 0)           { NEEDV(a); onnx_path = v; }
        else if (strcmp(a, "--instances-root") == 0) { NEEDV(a); instances_root = v; }
        else if (strcmp(a, "--device") == 0)         { NEEDV(a); device = v; }
        else if (strcmp(a, "--vehicles") == 0)       { NEEDV(a); vehicles = parse_ll(v, a);
                                                       has_vehicles = true; }
        else if (strcmp(a, "--max-iter") == 0)       { NEEDV(a); max_iter = parse_ll(v, a);
                                                       has_max_iter = true; }
        else if (strcmp(a, "--optimal") == 0)        { NEEDV(a); optimal = parse_f64(v, a);
                                                       has_optimal = true; }
        else if (strcmp(a, "--time-limit") == 0)     { NEEDV(a); time_limit = parse_f64(v, a);
                                                       has_time_limit = true; }
        else if (strcmp(a, "--no-diagnostics") == 0) emit_diag = false;
        else if (strcmp(a, "--quiet") == 0)          quiet = true;
        else { fprintf(stderr, "unknown option '%s'\n", a); return usage(stderr); }
#undef NEEDV
    }

    if ((size_preset == NULL) == (toml_path == NULL))
        sag_die("give exactly one of --size-preset or --config");
    if (!instance_arg && size_preset) sag_die("--size-preset mode requires --instance");
    if (recipe && !size_preset)
        sag_die("--recipe applies to --size-preset only; with --config set [tts] enabled in "
                "the TOML (false = the Onion-Sweep recipe)");
    if (recipe && strcmp(recipe, "full") != 0 && strcmp(recipe, "os") != 0)
        sag_die("--recipe must be 'full' or 'os', got '%s'", recipe);
    if (!onnx_path)
        sag_die("a forward backend is required: --onnx <model.onnx>");
    if (has_vehicles && vehicles <= 0)
        sag_die("--vehicles must be positive, got %lld (omit it to infer ceil(d(V)/Q))",
                (long long)vehicles);
    if (has_max_iter && max_iter < 0)
        sag_die("--max-iter does not accept a negative value (got %lld): in the reference a "
                "negative max_iter means 'finite and already reached' (a single bare LP round), "
                "whereas here it would mean 'unlimited'. Omit the option to run unlimited", (long long)max_iter);

    sag_config cfg;
    char root_buf[512] = "";
    {
        const char *er = getenv("SAGSEP_INSTANCES_ROOT");
        if (er) snprintf(root_buf, sizeof root_buf, "%s", er);
    }
    char toml_out_buf[512] = "";
    sag_toml_config tc;
    run_spec specs[SAG_TOML_MAX_INSTANCES];
    int64_t n_specs = 0;
    if (size_preset) {
        cfg = sag_config_preset(size_preset);
        if (recipe && strcmp(recipe, "os") == 0) cfg.tts.enabled = false;
        if (!out_dir) sag_die("--size-preset mode requires --out");
        specs[0].name = instance_arg;
        specs[0].vehicles = has_vehicles ? vehicles : -1;
        specs[0].max_iter = has_max_iter ? max_iter : -1;
        specs[0].optimal = optimal;
        specs[0].has_optimal = has_optimal;
        n_specs = 1;
    } else {
        sag_config_from_toml(toml_path, &tc);
        cfg = tc.cfg;
        if (tc.instances_root[0]) snprintf(root_buf, sizeof root_buf, "%s", tc.instances_root);
        if (tc.output_dir[0]) snprintf(toml_out_buf, sizeof toml_out_buf, "%s", tc.output_dir);
        if (instance_arg) {
            bool matched = false;
            for (int64_t i = 0; i < tc.n_instances; ++i) {
                if (strcmp(tc.instances[i].name, instance_arg) != 0) continue;
                matched = true;
                if (!has_vehicles) { vehicles = tc.instances[i].vehicles; has_vehicles = vehicles > 0; }
                if (!has_optimal && tc.instances[i].has_optimal) {
                    optimal = tc.instances[i].optimal;
                    has_optimal = true;
                }
                if (!has_max_iter && tc.instances[i].max_iter >= 0) {
                    max_iter = tc.instances[i].max_iter;
                    has_max_iter = true;
                }
                break;
            }
            if (!matched)
                sag_die("--instance '%s' is not in the [[instances]] list of %s (%lld entries). "
                        "Accepting it silently would make vehicles fall back to the inferred "
                        "value and produce a wrong lower bound that looks correct", instance_arg, toml_path,
                        (long long)tc.n_instances);
            specs[0].name = instance_arg;
            specs[0].vehicles = has_vehicles ? vehicles : -1;
            specs[0].max_iter = has_max_iter ? max_iter : -1;
            specs[0].optimal = optimal;
            specs[0].has_optimal = has_optimal;
            n_specs = 1;
        } else {
            if (has_vehicles || has_optimal)
                sag_die("--vehicles/--optimal are per-instance and are not accepted in "
                        "multi-instance mode; to override, pass --instance <name> and run that one");
            if (tc.n_instances <= 0)
                sag_die("%s contains no [[instances]] entries", toml_path);
            for (int64_t i = 0; i < tc.n_instances; ++i) {
                specs[i].name = tc.instances[i].name;
                specs[i].vehicles = tc.instances[i].vehicles;
                specs[i].max_iter = has_max_iter ? max_iter : tc.instances[i].max_iter;
                specs[i].optimal = tc.instances[i].optimal;
                specs[i].has_optimal = tc.instances[i].has_optimal;
            }
            n_specs = tc.n_instances;
        }
        if (!out_dir && !toml_out_buf[0])
            sag_die("no output directory: pass --out, or set output_dir in %s", toml_path);
        if (!out_dir) out_dir = toml_out_buf;
    }
    if (instances_root) snprintf(root_buf, sizeof root_buf, "%s", instances_root);
    if (has_time_limit) cfg.lp.time_limit = time_limit;
    sag_config_check_frozen(&cfg);

    ensure_dir(out_dir);

    sag_arena inst_arena, result_arena;
    sag_arena_init(&inst_arena, 512u << 20, "cli.instance");
    sag_arena_init(&result_arena, 512u << 20, "cli.result");

    sag_forward *shared_forward = NULL;
    {
        const char *ep = device ? device : cfg.model.device;
        if (!ep || !*ep) ep = "cpu";
        if (strncmp(ep, "cuda", 4) == 0) {
            fprintf(stderr,
                    "[sagsep] WARNING: policy_device='%s' -- using the GPU forward pass. "
                    "GPU reduction order is not fixed, so probs are not bit-for-bit reproducible; "
                    "pass --device cpu explicitly for reproducible runs.\n", ep);
            fflush(stderr);
        }
        shared_forward = sag_forward_open_onnx(onnx_path, ep);
    }

    for (int64_t si = 0; si < n_specs; ++si) {
        const run_spec *sp = &specs[si];
        if (n_specs > 1) {
            fprintf(stderr, "[sagsep] ==== instance %lld/%lld: %s ====\n",
                    (long long)(si + 1), (long long)n_specs, sp->name);
            fflush(stderr);
        }
        sag_arena_reset(&inst_arena);
        sag_arena_reset(&result_arena);

        char vrp_path[1024], stem[SAG_NAME_MAX];
        resolve_vrp_path(vrp_path, sizeof vrp_path, sp->name, root_buf);
        instance_stem(stem, sizeof stem, sp->name);

        sag_instance inst;
        sag_load_instance(&inst_arena, vrp_path, sp->vehicles > 0 ? sp->vehicles : 0,
                          sp->has_optimal, sp->optimal, &inst);
        if (sp->vehicles <= 0) inst.vehicles = infer_vehicles(&inst);

        sag_loop_options opt;
        sag_loop_options_init(&opt);
        opt.max_iter = sp->max_iter >= 0 ? sp->max_iter : -1;
        opt.emit_diagnostics = emit_diag;
        opt.progress = quiet ? NULL : stdout;

        opt.forward = shared_forward;

        printf("[sagsep] arm=learned size=%s recipe=%s\n", cfg.size[0] ? cfg.size : "(toml)",
               cfg.tts.enabled ? "full (sweep+onion+tts)" : "os (sweep+onion, tts off)");
        printf("[sagsep] instance = %s (N=%lld K=%lld Q=%lld)\n", inst.name,
               (long long)inst.n_nodes, (long long)inst.vehicles, (long long)inst.capacity);
        printf("[sagsep] forward = %s\n", sag_forward_describe(opt.forward));
        printf("[sagsep] gurobi Method=%lld Threads=%lld time_limit=%g\n",
               (long long)cfg.lp.gurobi_method, (long long)cfg.lp.gurobi_threads,
               cfg.lp.time_limit);
        fflush(stdout);

        sag_root_result res;
        sag_run_instance(&inst, &cfg, &opt, &result_arena, &res);

        printf("------------------------------------------------------------\n");
        printf("Instance : %s\n", stem);
        printf("Vehicles : %lld\n", (long long)inst.vehicles);
        if (inst.has_optimal) printf("Optimal  : %.2f\n", inst.optimal);
        printf("Lower bd : %.2f\n", res.lower_bound);
        printf("Runtime  : %.2f s\n", res.runtime);
        if (res.time_limit_hit) printf("Note     : stopped after hitting the time limit\n");
        printf("Cut iters: %lld (total cuts %lld, last added %lld cuts, max violation %.4f)\n",
               (long long)res.iteration_count, (long long)res.total_cuts,
               (long long)res.json.last_cuts_added, res.json.last_max_violation);
        sag_json_write_result_file(out_dir, stem, &res.json);
        printf("Stored at: %s/root_relax_%s_learned_fb.json\n", out_dir, stem);
        fflush(stdout);

#if defined(__GLIBC__)
        malloc_trim(0);
#endif
    }

    if (shared_forward) sag_forward_close(shared_forward);
    sag_arena_destroy(&result_arena);
    sag_arena_destroy(&inst_arena);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) return usage(stderr);
    const char *cmd = argv[1];
    if (strncmp(cmd, "--", 2) == 0) {
        if (strcmp(cmd, "--help") == 0) { (void)usage(stdout); return 0; }
        return cmd_run(argc - 1, argv + 1);
    }
    if (strcmp(cmd, "preset") == 0)  return cmd_preset(argc - 2, argv + 2);
    if (strcmp(cmd, "build") == 0)   return cmd_build();
    if (strcmp(cmd, "vrp") == 0)     return cmd_vrp(argc - 2, argv + 2);
    if (strcmp(cmd, "-h") == 0) { (void)usage(stdout); return 0; }
    fprintf(stderr, "unknown command '%s'\n", cmd);
    return usage(stderr);
}
